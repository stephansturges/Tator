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

In order to run the API which serves this "auto mode" you will need to first load a YOLO-format dataset into the tool, export the dataset in the right format (one button) and then run a python script to train the "auto mode" which is just a logistic regression on top of CLIP. 
Then you can run the API / backend which will serve your CLIP + regression as "auto class annotation" and will also serve SAM in bounding-box or point-mode. Once you have these tools combined you can annotate your data in a single click, letting SAM give you a clean boundinx box and CLIP giving you a class. Of course this won't work 100% of the time but in little experiments I get to >95% accuracy with very little data, which speeds up any remaining manual annotation by a LOT!

## Let's get started!

I'm assuming you've installed all dependencies. I need to write how to do that and add a requirements.txt, for now that's in the "to do" pile.

OK so let's train the CLIP logistic regression.

For this you will need a set of data in YOLO format, and you'll need to open the ybat.html file in ./ybat-master/ in your browser.

Next, load a set of labeled images, their labelmap and their bounding boxes in the main tool, then click "crop&save", then wait for the .zip to be exported. 

Save the .zip in your root file of this tool (next to clip_kmeans_and_regress.py) and unzip it so that the contents are in ./crops/

Now run python3 clip_kmeans_and_regress.py, which will take a while depending on your machine. This will train a logistic regression on top of the CLIP vectors extracted from all the little images that are now in your ./crops/ folder, and save it as a .pkl file in the root of this project.

Once it's done you will have 2 files called my_label_list.pkl and my_logreg_model.pkl, and it will display some accuracy statistics on your label-based clustering.

Now you're ready to fire up the backend which will serve your CLIP-based label-automation and SAM. You can do this by going to the terminal and running: 

uvicorn localinferenceapi:app --host 0.0.0.0 --port 8000

Assuming you have all the dependencies set up you should see some output in the console like: 
Loading CLIP model...
Loading logistic regression...
INFO:     Started server process [79641]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


You're good to go, time to go back to your browswer!

Now if you select "Auto Mode" in the interface before creating a bounding box your local CLIP API which will serve you an automatic label guess for the bbox you are creating. 

If the guess is wrong, remove "auto mode" and make the bbox manually.

Messing around a bit on some of my local datasets I can get >95% confidence in the CLIP suggestions, which makes manual annotation a LOT easier. 

You can also use "SAM mode" (which needs renaning to something like "SAM bbox mode") to create a bounding box with SAM: just drag a bounding box and wait a second, and SAM will refine it for you.
You can also use "Point mode", which uses SAM with a single "spot" annotation to suggest a bounding box.

Each one of these SAM-based bbox-suggestion-modes can ALSO be combined with "auto mode", in which case you get the bbox from SAM and the class label from CLIP, for lighting-fast new-data annotation!

For now there are some bits that need cleaning up in the interface: bad names for tools, missing tooltips and keyboard shortcuts, and I need to make the "SAM mode" (bbox or point) a radial since you cannot use both at the same time... but it already works quite nicely :) 


## todo

Need a proper requirements file, especially explaining how to install CLIP
Need setup instructions
Add shortcuts for missing modes
Remove Pascal VOC export
Make SAM modes into radial
Rename the 2 SAM modes to make more sense
Add option in API to run without regression .pkl + train regression from API 
Make requirements.txt + add install explanations (especially CLIP)

## Thanks

Thanks to @drainingsun for a nice clean tool with YBAT!
Thanks to OpenAI and Meta for the amazing free tools :) 


## Licenses

License for all third-party packages including Clip (OpenAI) and SAM (Meta) are the users' own responsibility.

YBAT code license can be found above, and you can find the original repository here: https://github.com/drainingsun/ybat

All the novel code is MIT license:

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
