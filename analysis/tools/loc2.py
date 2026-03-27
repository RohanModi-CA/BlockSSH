"""
We're going to start looking at localization in a proper way.
This will be called through a viz/, of course, but 
the calling will go like python3 viz/site_amplitudes.py dataset.json peaks.csv.

There are arguments to the viz script (not implemented for this
tools script, but just so we know what we need to make easy):

--integration_window_width=0.1
--normalization_multiplier=4
--preview

--phase_reconstruction


There are going to be two ways we do this. The first is the least brittle.
In this case, which is the default way it's done, we look at all the nondisabled
datasets. We first read in the CSV of peaks. We assert that they're sorted least to greatest if not assertionerror or otherwise error. 

In these nondisabled datasets, look at which global bond indices they contain. We get a dictionary keyed by global bond index and which contains another dictionary, which just contains: dataset:str, and local_index:int. local_index is the index in the dataset corresponding to this global bond index. This should take into account disabled indices. 
Then, for each bond index, we load in their FFTs. We average their respective FFTs, to get one FFT for each bond index. We're looking at the *ampltiudes*, not power or complex transform.

Now in a function, we process this averaged FFT. We mask the following range, the range_of_interest: lowerbound: (csvpeaks[0]-(integration_window_width*normalization_multiplier)),
upperbound: (csvpeaks[-1]+(integration_window_width*normalization_multiplier). If this crosses the edge of a dataset, raise an error and exit. Now, first thing is we scipy.signal.detrend(linear) on this range. Then we integrate this masked range of ampltiudes to get the area under the amplitudes. Then, multiply this averaged and masked FFT by 1/integral such that we've normalized wrt this area. This masked FFT should now be attached to the dictionary of this bond index for organization.

Then, there's another function. This function takes the big global bonds dictionary. For each bond, it calls a function and passes that bonds dictionary to it. This function looks at the peaks. It calls a function on each peak. It goes to each peak from the CSV(which should be taken as a variable, the parsed list from the CSV, that is). It integrates the amplitude between (peak-integration_window_width) and (peak+integration_window_width). We then make a tuple: (peak_hz, integrated_ampltiude). Finally we end up with a list of tuples. This gets added to the dictionary for this global bond. We run this on all the global bonds.

Ultimately, we return a dictionary with the global bond indices and the list of tuples.

The other case is the --phase_reconstruction case. We are not going to currently implement this, and we should just leave that space blank, perhaps in an elif: not implemented.



"""
