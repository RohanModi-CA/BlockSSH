Hi Defeng, here are some things about how the code works and how to use it.

---
So the basic directory structure is like this:

```
gram@gram:~/Documents/FileFolder/minefield/block25/new_3 (rohan)$ tree -L 1
.
├── analysis
├── check_results
├── figures
├── helper
├── other
├── __pycache__
├── theory
├── track

```


Note: I'm assuming you're using a bash-like terminal, which you really should be... (So, not Windows Command Prompt).. use Git Bash or something else. I think VSCode can use Git Bash if you specify. When you're going to start working, you should activate the environment by calling:

```
source ./activate
```

This, hopefully, will enable tab completion to make calling things easier.

# Tracking
---

So first thing, the track happens in the `track/` folder. There's a lot going on here, so the basic process is like this:


1. Move the video into the `track/Videos` folder. 
	1. Name the video as you want, `MYDATASET.mp4` (or .mov or whatever). `MYDATASET` is how we're going to pass this to further scripts. 

## Prepare for Tracking

1. Next we start the tracking process. For the new black videos from the bottom, this is done in `track/Bottom/`, and `track/Front` has analagous things from before. The first step is that we need to **prepare the video** for tracking. 
	1. This is done with `track/Bottom/0.VideoPrepareBottom.py`. Call it like:

```bash
python3 0.VideoPrepareBottom.py MYDATASET
```

Here you'll set the crops on the video, trim the video in time, set the limited rotation, and check that it tracks correctly. If it does not, you should edit the tracking parameters, as it prompts you to. The prepare step does not rewrite the video; it only saves the tracking parameters.

_ENSURE YOU ROTATE THE VIDEO, SUCH THAT THE LONGITUDINAL AXIS IS HORIZONTAL (X)_.

Once everything tracks and you're happy with it, let it save the tracking parameters. 

## Tracking

So now that you've set the tracking parameters you can start the tracking.

```py
python3 1.TrackRun.py MYDATASET
```
And let that go work.

# Processing & Verification

This is the scariest step because if the tracking didn't work very well it can become very annoying to recover... so first thing, run the processor and hope it passes verification.

```py
python3 2.ProcessVerify.py MYDATASET
```

This will go in & check that the dataset is not broken and if it's not it will process it into our data format. _If it is broken,_ it can try to fix it for you, say yes, please fix it, and cross your fingers that it works. If auto-repair is not enough, it will exit and tell you to do the manual repair.

If it works, great, we're done with tracking! if not.... this is very annoying; you have to manually fix it:
### If broken...

First see why it is broken.
```
python3 track/debug_overlay_track1.py MYDATASET
```

It'll show you a couple good frames then the bad frames (click the buttons it says on the screen to navigate).

It's probably missing detections, so you can go in and manually fix it...

```
python3 2b.ManualRepair.py MYDATASET
```

And from here you can manually add in extra detections, clicking the buttons it says to click, trying to place them appropriately and placing the angle appropriately; it doesn't have to be perfect; it'll just be a tiny bit of noise which will just average out; the important thing is just not having delta functions or the (I imagine) much bigger noise from haphazard linear interpolation..

Then you should try to verify it again and hope it passes verification.

```
python3 2.ProcessVerify.py MYDATASET
```

---

#### Labeling

So the final thing we have to do is label everything. So we're going to run

```
python3 3.Label.py MYDATASET
``` 

Here you will identify which site is which. Make sure you 1-index: the first block/site from the left is block/site 1. The downstream code derives the bonds from the site labels.

Often some of the edge tracking data is garbage, so we can also disable the edge sites. 

---


So now once this is done, we're done with tracking! So this creates a folder in `track/data` named `MYDATASET`, which contains the processed data in this layout:

```
track/data/MYDATASET/
├── manifest.json
├── params_bottom.json
├── track1.msgpack
├── labels.json
└── components
    ├── x/track2_permanence.msgpack
    ├── y/track2_permanence.msgpack
    └── a/track2_permanence.msgpack
```

You shouldn't have to deal with any of these, but to share data with me you can just send me this dataset folder zipped up, and vice versa, you can place this folder manually in `track/data` if I send you it.


# Analysis
---
So now all the analysis you'll do is done within `analysis/go/`.

Here are the different files and functions.

#### MakeGroup.py

This just defines a group of datasets which can be convenient for later.

```
python3 MakeGroup.py MYDATASET1 MYDATASET2 MYDATASET3 MYGROUP
```


#### Timeseries.py

This plots the timeseries of all bonds in every component. 

```
python3 Timeseries.py MYDATASET
```

`--not-bonds` shows the raw timeseries per site, not per bond.
#### FFT.py

This shows you the FFTs of any all the bonds and components within a dataset.

```
python3 FFT.py MYDATASET
```

You can save this FFT to a `spectrasave` if you want to look at peaks, for example. 
```
python3 FFT.py MYDATASET --spectrasave SPECTRASAVENAME
```

You can also look at a group of datasets averaged together.

```
python3 FFT.py --group MYGROUP
```

#### Subtract.py

This is how we subtract different components from others. This can take either a single dataset or a group of datasets.

```
python3 Subtract.py MYDATASET
```

or

```
python3 Subtract.py --group MYGROUP
```

By default this assumes you want to look at the `x` (longitudinal data), but you can change that with `--show y` or `--show a`. 

By default this subtracts both the angle data and the y data at scaling factor 1. You can also manually say what to subtract, with `--subtract y 0.5 --subtract a 1.8`. 

By default if you have more than one subtraction, as we do by default, this also subtracts the following subtractions to avoid double subtraction. `analysis/go/Subtract.py` does not currently expose `--secondary-subtract-ratio`.

By default this will show the others, you can disable that with `--only-result`.

If you want to save this plot to look at peaks you can do:

```
python3 Subtract.py MYDATASET --spectrasave SPECTRASAVENAME
```

Note: this will only save the final subtracted spectrum, not any of the intermediates shown.

#### ClickPeakFind.py

This is how I set peaks, lol, because it is the easiest. (I also used a click-peak-finder for the laser lab in 469) This takes a spectrasave.

```
python3 ClickPeakFind.py SPECTRASAVENAME --csv ../configs/peaks/PEAKSNAME.csv
```

Here `PEAKSNAME` is the name of this group of peaks.

Then the process is a bit annoying to do this because matplotlib is not really meant for this but basically you have to enter 'add' mode to add a peak, remove to remove, etc. I think you can figure out how that works. You can also run it again, with the same PEAKSNAME, to edit or view that peaks file. 
#### SpectrasaveView.py

This just loads a spectrasave and shows it to you.

```
python3 SpectrasaveView.py SPECTRASAVENAME
```

#### Wavefunctions.py

This takes a single dataset with multiple bonds and then plots the wavefunctions, with respect to peaks you've previously defined with ClickPeakFind.

```
python3 Wavefunctions.py MYDATASET PEAKSNAME
```

You can do `--forcereal` to force the phases to be purely real but I think this gives worse results. You can do `--flip 1 --flip 8` to multiply bonds 1 and 8 wavefunctions by negative one, or `--flip all`. We're free to do this as we want. 

We get this through integration, but you could also do `--highest-bin` for that approach, but that I think is worse. 

You can click on the different bonds to see what the peak looks like on that bond and the integration window. You can change the integration window with `--integration-window`.


