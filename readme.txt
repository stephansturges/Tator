🥔

## my little YOLO annotator

This is my little Yolo annotation tool. It's super light and simple and runs locally but uses some nice bits like CLIP and SAM on the backend to auto-annotate and make things super easy to use. I rarely label stuff manually but when I do I get *super frustrated* because offline tools are slooooow and clunky and online tools are laggy and expensive and want to force you into contorted workflows and upsell stuff... which is probably great for a big enterprise, but when I'm doing manual annotation it's usually because I'm just messing around with a new dataset or want to benchmark some stuff and I don't need a 1-year 5-seat license with 20 credits to train models.


Most of the frontend code is taken from YBAT, which can be found here:
https://github.com/drainingsun/ybat

See the original YBAT license here:
https://github.com/drainingsun/ybat/blob/master/LICENSE

The backend usese CLIP from OpenAI (https://github.com/openai/CLIP) and SAM from Meta (https://github.com/facebookresearch/segment-anything) to automate annotation in a super-responsive and FAST way. 

## How does it work?

Basically it works like YBAT, but I just added a couple of interface changes to change some colors and allow "auto mode". 

## Auto mode with Clip!

Auto mode uses clustering with CLIP to suggest classes for objects automatically as you annotate. Note if the automatic suggestion is wrong you just need to turn off "auto mode" to label manually.

You need to first train your CLIP model + logistic regression, however!

In order to do this, load a set of labeled images, their labelmap and their bounding boxes in the main tool, then clip "crop&save", then wait for the .zip to be exported. 
Save the .zip in your root file (next to clip_kmeans_and_regress.py) and unzip it so that the contents are in ./crops/
Run python3 clip_kmeans_and_regress.py, which will take a while depending on your machine. It will generate embeddings for all the crops and train a little logistic regression on top, saving it as .pkl file. It also saves a labelmal as .pkl
Once you have these files, you can run:
uvicorn localinferenceapi:app --host 0.0.0.0 --port 8000

... and leave the API running. 

Now if you select "Auto Mode" in the interface a crop will automatically get sent to the local CLIP API which will serve you an automatic label guess for the bbox you are creating. If the guess is wrong, remove "auto mode" and make the bbox manually.

Messing around a bit on some of my local datasets I can get >95% confidence in the CLIP suggestions, which makes manual annotation a LOT easier. 

I would still recommend you mostly stick with synthetic data if you want to do things quickly, but that's a different subject ...


## todo

Need a proper requirements file, especially explaining how to install CLIP
Need setup instructions


## Thanks

Thanks to @drainingsun for a nice clean tool with YBAT!


## Licenses

License for all third-party packages including Clip (OpenAI) are the users' own responsibility.

YBAT code license can be found above.

Anything added to the YBAT code is also MIT license with the license below:

Copyright (c) 2025 Aircortex.com

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
