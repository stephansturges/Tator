"""Bound request bytes even when clients omit Content-Length."""
from starlette.responses import JSONResponse


class RequestBodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    def __init__(self, app, json_limit=17 * 1024 * 1024, upload_limit=260 * 1024 * 1024):
        self.app, self.json_limit, self.upload_limit = app, json_limit, upload_limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        upload = headers.get(b'content-type', b'').lower().startswith(b'multipart/form-data')
        limit = self.upload_limit if upload else self.json_limit
        try:
            declared = int(headers.get(b'content-length', b'0'))
        except ValueError:
            return await JSONResponse({'error': 'invalid_content_length'}, 400)(scope, receive, send)
        if declared < 0 or declared > limit:
            return await JSONResponse({'error': 'request_too_large'}, 413)(scope, receive, send)
        consumed, rejected, started = 0, False, False

        async def bounded_receive():
            nonlocal consumed, rejected
            message = await receive()
            if message['type'] == 'http.request':
                consumed += len(message.get('body', b''))
                if consumed > limit:
                    rejected = True
                    raise RequestBodyTooLarge()
            return message

        async def bounded_send(message):
            nonlocal started
            if rejected:
                # Parsers sometimes convert receive exceptions into a generic
                # malformed-body response. Preserve the actual size-limit cause.
                return
            if message['type'] == 'http.response.start':
                started = True
            await send(message)
        try:
            await self.app(scope, bounded_receive, bounded_send)
        except Exception:
            if not rejected:
                raise
        if rejected and not started:
            await JSONResponse({'error': 'request_too_large'}, 413)(scope, receive, send)
