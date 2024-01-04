import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Set font default
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
matplotlib.rcParams['mathtext.rm'] = 'stix'
matplotlib.rcParams['mathtext.it'] = 'stix'
matplotlib.rcParams['mathtext.bf'] = 'stix'

plt.rcParams['font.family'] = 'serif'  # or 'DejaVu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 'DejaVu Serif' 'serif' 'Times
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
'''

tickfontsize = 30
fontsize = 30
linewidth = 4
markersize = 15

SMALL_SIZE = tickfontsize
MEDIUM_SIZE = tickfontsize
BIGGER_SIZE = fontsize

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

shapes = ['o','s','^','D','*']

# Get the errors for each data size
data_sizes = [10,50,250,1000,2000, 4000, 6000, 8000, 9500]
f2f_errors = np.ones((5, 9))
f2v_errors = np.ones((5, 9))
v2f_errors = np.ones((5, 9))
v2v_errors = np.ones((5, 9))

err_type = 'Abar_abs_error_mean'
for sample in range(5):
    for datasize in data_sizes:
        if datasize >9000:
            pass
        else:
            f2f_error_file = '/groups/astuart/mtrautne/FNM/FourierNeuralMappings/homogenization/trainModels/trainedModels/size_compare/vor_data_'+ str(datasize) + '_' + str(sample) + '_new_errors.yml'
            # read error file and extract Abar error
            with open(f2f_error_file, 'r') as f:
                    error_dict = yaml.load(f, Loader=yaml.FullLoader)
                    f2f_errors[sample, data_sizes.index(datasize)] = error_dict[err_type]
        f2v_error_file = '/groups/astuart/mtrautne/FNM/FourierNeuralMappings/homogenization/trainModels/trainedModels/size_compare' + '/vor_f2v_data_size_' + str(datasize) + '_' + str(sample) + '_errors.yml'
        # read error file and extract Abar error
        try:
            with open(f2v_error_file, 'r') as f:
                error_dict = yaml.load(f, Loader=yaml.FullLoader)
                f2v_errors[sample, data_sizes.index(datasize)] = error_dict[err_type]
        except:
            print('data size: ', datasize, ' sample: ', sample)
            f2v_errors[sample, data_sizes.index(datasize)] = f2v_errors[sample-1, data_sizes.index(datasize)]

        v2f_error_file = '/groups/astuart/mtrautne/FNM/FourierNeuralMappings/homogenization/trainModels/trainedModels/size_compare' + '/vor_v2f_data_size_' + str(datasize) + '_' + str(sample) + '_new_errors.yml'
        v2v_error_file = '/groups/astuart/mtrautne/FNM/FourierNeuralMappings/homogenization/trainModels/trainedModels/size_compare' + '/vor_v2v_data_size_' + str(datasize) + '_' + str(sample) + '_errors.yml'
        # read error file and extract Abar error
        with open(v2f_error_file, 'r') as f:
            error_dict = yaml.load(f, Loader=yaml.FullLoader)
            v2f_errors[sample, data_sizes.index(datasize)] = error_dict[err_type]
        with open(v2v_error_file, 'r') as f:
            error_dict = yaml.load(f, Loader=yaml.FullLoader)
            v2v_errors[sample, data_sizes.index(datasize)] = error_dict[err_type]    

# Add the errors for the 9500 data size
for sample in range(5):
    vor_error_file = '/groups/astuart/mtrautne/FNM/FourierNeuralMappings/homogenization/trainModels/trainedModels/size_compare/vor_data_9500_' + str(sample) + '_new_errors.yml'
    # read error file and extract H1 mean error
    with open(vor_error_file, 'r') as f:
            error_dict = yaml.load(f, Loader=yaml.FullLoader)
            f2f_errors[sample, -1] = error_dict[err_type]

# Get the mean and std of the errors
f2f_mean_errors = np.mean(f2f_errors, axis=0)
f2f_2std_errors = 2*np.std(f2f_errors, axis=0)
f2v_mean_errors = np.mean(f2v_errors, axis=0)
f2v_2std_errors = 2*np.std(f2v_errors, axis=0)
v2f_mean_errors = np.mean(v2f_errors, axis=0)
v2f_2std_errors = 2*np.std(v2f_errors, axis=0)
v2v_mean_errors = np.mean(v2v_errors, axis=0)
v2v_2std_errors = 2*np.std(v2v_errors, axis=0)

print(v2f_mean_errors)
print(v2f_2std_errors)

# Plot the results
plt.figure(figsize=(10,10))
plt.plot(data_sizes, f2f_mean_errors, label='F2F',marker = shapes[1],color = CB_color_cycle[0],linewidth=linewidth,markersize=markersize)
plt.errorbar(data_sizes, f2f_mean_errors, yerr=f2f_2std_errors,color = CB_color_cycle[0])
plt.fill_between(data_sizes, f2f_mean_errors - f2f_2std_errors, f2f_mean_errors + f2f_2std_errors, alpha=0.2, color = CB_color_cycle[0])
plt.plot(data_sizes, f2v_mean_errors, label='F2V',marker = shapes[0],color = CB_color_cycle[1],linewidth=linewidth,markersize=markersize)
plt.errorbar(data_sizes, f2v_mean_errors, yerr=f2v_2std_errors,color = CB_color_cycle[1])
plt.fill_between(data_sizes, f2v_mean_errors - f2v_2std_errors, f2v_mean_errors + f2v_2std_errors, alpha=0.2, color = CB_color_cycle[1])
plt.plot(data_sizes, v2f_mean_errors, label='V2F',marker = shapes[2],color = CB_color_cycle[2],linewidth=linewidth,markersize=markersize)
plt.errorbar(data_sizes, v2f_mean_errors, yerr=v2f_2std_errors,color = CB_color_cycle[2])
plt.fill_between(data_sizes, v2f_mean_errors - v2f_2std_errors, v2f_mean_errors + v2f_2std_errors, alpha=0.2, color = CB_color_cycle[2])
plt.plot(data_sizes, v2v_mean_errors, label='V2V',marker = shapes[3],color = CB_color_cycle[3],linewidth=linewidth,markersize=markersize)
plt.errorbar(data_sizes, v2v_mean_errors, yerr=v2v_2std_errors,color = CB_color_cycle[3])
plt.fill_between(data_sizes, v2v_mean_errors - v2v_2std_errors, v2v_mean_errors + v2v_2std_errors, alpha=0.2, color = CB_color_cycle[3])

plt.legend(fontsize=fontsize)
plt.xlabel('Number of Training Samples',fontsize=fontsize)
plt.ylabel(r'Mean Absolute $\overline{A}$ Error',fontsize=fontsize)
# set xticks
plt.xticks(data_sizes,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
#log scales
plt.yscale('log')
plt.xscale('log')
# get log-log slope
x = np.log(data_sizes)
y = np.log(f2f_mean_errors)
m, b = np.polyfit(x, y, 1)
print('F2F slope: ', m)
# add f2f slope to plot
# plt.text(0.1,0.1,r'F2F slope: ' + str(round(m,2)),fontsize=fontsize,transform=plt.gca().transAxes)

y = np.log(f2v_mean_errors)
m, b = np.polyfit(x, y, 1)
print('F2V slope: ', m)
# add f2v slope to plot
# plt.text(0.1,0.65,r'F2V slope: ' + str(round(m,2)),fontsize=fontsize,transform=plt.gca().transAxes)
plt.title('QQQ', color = 'white',fontsize=fontsize)
# set xticks, not scientific notation
# remove extra xticks

y = np.log(v2f_mean_errors)
m, b = np.polyfit(x, y, 1)
print('V2F slope: ', m)

y = np.log(v2v_mean_errors)
m, b = np.polyfit(x, y, 1)
print('V2V slope: ', m)

# make not scientific notation
# Plot a line with slope -1/2
x = np.linspace(10,9500,100)
y = 0.5*x**(-1/2)
plt.plot(x,y,linestyle='--',color='black',linewidth=linewidth)
plt.text(0.3,0.6,r'$\mathcal{O}(N^{-1/2})$',fontsize=fontsize,transform=plt.gca().transAxes)

# reverse legend order
handles, labels = plt.gca().get_legend_handles_labels()
order = [3,1,2,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=fontsize)

plt.savefig('data_size_mean_compare.pdf',bbox_inches='tight')
