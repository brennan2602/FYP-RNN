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


def encode(arr):
    timeinc=0
    outString=""
    for time in arr:
        notesinc = -1
        if np.all(time==0):
            outString=outString+"#"
        for vel in arr[timeinc]:
            notesinc=notesinc+1
            if vel != 0:
                noteRep=str(notesinc)+"-"+ str(vel)+"_"
                noteRep="("+str(notesinc)+","+ str(vel)+")"
                outString=outString+noteRep
        outString=outString+"\n"
        timeinc = timeinc+1
    return outString
files=glob.glob(r".\dataset\train\*.midi")
print(files)
for f in files[0:5]:
    x= f.split("\\")[-1]
    print(x)
    fileName=f+"\\"+x
    # with open(fileName) as file:
    #     content = file.read()
    pr = get_piano_roll(fileName)
    arr = pr.T
    outString= encode(arr)
    file1 = open("data.txt","a")
    file1.write(outString)

file1.close()