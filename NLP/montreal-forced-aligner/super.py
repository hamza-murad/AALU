import textgrid
import subprocess
subprocess.call('D:\\montreal-forced-aligner\\bin\\mfa_align D:\\montreal-forced-aligner\\input D:\\montreal-forced-aligner\\english_dict.txt D:\\montreal-forced-aligner\\english.zip D:\\montreal-forced-aligner\\output')

data = textgrid.TextGrid.fromFile('D:\\montreal-forced-aligner\\output\\input\\test.TextGrid')
ls = data.getList('phones')
D = {}
for i in ls[0]:
    D[i.mark] = (i.minTime, i.maxTime)

print(D)
with open('phonemesFile.txt', 'w') as f:
    for item in ls[0]:
        f.write("%s\n" % item)