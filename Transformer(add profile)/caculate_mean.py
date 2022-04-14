path = 'D:/PETER2/MT_output/rank_profile_transformer_fea_256/HIT'
import os
import re
import numpy as np
files = os.listdir(path)
def file_filter(f):
    if f[-4:] in ['.log']:
        return True
    else:
        return False

files = list(filter(file_filter, files))
fs = []
write_path = os.path.join(path,'mean_result.txt')
w = open(write_path,'w')
metrcs = ['on test', 'RMSE', "MAE", "BLEU-1", "BLEU-4", 'USR','DIV','FCR','FMR','rouge_1/f_score','rouge_1/r_score','rouge_1/p_score','rouge_2/f_score','rouge_2/r_score','rouge_2/p_score']
metrcs_dict=dict()
files_path = [os.path.join(path,i) for i in files]
for i in files_path:
    f = open(i,'r')
    f = f.readlines()[50:]
    for m in metrcs:
        for n in f:
            if m in n:
                if m in metrcs_dict:
                    n = n.split(':')[-1]
                    # metrcs_dict[m].append([float(s) for s in re.findall(r'-?\d+\.?\d*', n)])
                    metrcs_dict[m].append([float(temp) for temp in n.split() if temp.replace('.','').isdigit()])
                else:

                    n = n.split(':')[-1]
                    metrcs_dict[m] = [[float(temp) for temp in n.split() if temp.replace('.','').isdigit()]]
metrcs_dict['USR'] = [[i[0]]for i in metrcs_dict['USR']]
mean_dict=dict()
for i in metrcs_dict:
    if i == 'on test':
        loss = np.array(metrcs_dict[i])
        mean_dict['context loss'] = np.mean(loss,0)[0]
        mean_dict['text loss'] = np.mean(loss,0)[1]
        mean_dict['rating loss'] = np.mean(loss,0)[2]
    else:
        mean_dict[i]=np.mean(np.array(metrcs_dict[i]))

# w.write('context loss'+' '+str(mean_dict['context loss'])+'\n')
# w.write('text loss'+' '+str(mean_dict['text loss'])+'\n')
# w.write('rating loss'+' '+str(mean_dict['rating loss'])+'\n')
w.write('RMSE'+' '+str(mean_dict['RMSE'])+'\n')
w.write('MAE'+' '+str(mean_dict['MAE'])+'\n')
w.write('FMR'+' '+str(mean_dict['FMR'])+'\n')
w.write('FCR'+' '+str(mean_dict['FCR'])+'\n')
w.write('DIV'+' '+str(mean_dict['DIV'])+'\n')
w.write('USR'+' '+str(mean_dict['USR'])+'\n')
w.write('BLEU-1'+' '+str(mean_dict['BLEU-1'])+'\n')
w.write('BLEU-4'+' '+str(mean_dict['BLEU-4'])+'\n')
w.write('rouge_1/p_score'+' '+str(mean_dict['rouge_1/p_score'])+'\n')
w.write('rouge_1/r_score'+' '+str(mean_dict['rouge_1/r_score'])+'\n')
w.write('rouge_1/f_score'+' '+str(mean_dict['rouge_1/f_score'])+'\n')
w.write('rouge_2/p_score'+' '+str(mean_dict['rouge_2/p_score'])+'\n')
w.write('rouge_2/r_score'+' '+str(mean_dict['rouge_2/r_score'])+'\n')
w.write('rouge_2/f_score'+' '+str(mean_dict['rouge_2/f_score'])+'\n')
w.flush()
w.close()