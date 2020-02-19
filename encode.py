import numpy as np
import pretty_midi
import glob

def get_piano_roll(midifile):
	#midi_data = pretty_midi.PrettyMIDI('test.midi')
	midi_pretty_format = pretty_midi.PrettyMIDI(midifile)
	piano_midi = midi_pretty_format.instruments[0] # Get the piano channels
	piano_roll = piano_midi.get_piano_roll(fs=25)
	print(piano_roll.shape)
	return piano_roll


# encoding both notes and velocity into same string
def encode(arr):
    timeinc=0 #time increment
    outString="" #string initially blank
    for time in arr: #looping through the columns of the piano roll array
        # (actually looping through rows of transposed piano roll so practically the same)
        notesinc = -1 #initialising note increment to -1 will almost immediately be updated to 0
        if np.all(time==0):#if everything in this time increment is 0 write a # to the encoded string (nothing playing)
            outString=outString+"#"
        for vel in arr[timeinc]: #loops through the current time increment (array of velocities)
            notesinc=notesinc+1 #matching the index for array at this time increment to a note
            # so notes and Vel can be grouped together
            if vel != 0: #if Vel!=0 a note is being played so add a note/velocity pair to the encoded file
                noteRep="("+str(notesinc)+","+ str(vel)+")" #this is the combined encoding format
                outString=outString+noteRep
        outString=outString+"\n" #now on a new time increment so move to a new line in encoded file
        timeinc = timeinc+1 #incrementing the sample that is currently being encoded
    return outString

files=glob.glob(r".\dataset\train\*.midi") #getting midi files from train split of MAESTRO dataset
print(files)

for f in files:# [0:1]: this will loop through and encode the whole dataset
    #some manipuation of strings due to containing folder and file having the same name
    x= f.split("\\")[-1]
    print(x)
    fileName=f+"\\"+x
    pr = get_piano_roll(fileName)
    arr = pr.T
    outString= encode(arr)
    file1 = open("encodedData.txt","a")
    file1.write(outString)

file1.close()

# Old code to read a file in directly kept here in case user ever wishes to do something similar

# file="OldFileforTraining.midi"
# pr=get_piano_roll(file)
# arr=pr.T
# outString=encode(arr)
# file1 = open("encodedAltData.txt", "a")
# file1.write(outString)
# file1.close()