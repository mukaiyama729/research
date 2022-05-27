from re import A
import pandas as pd
import numpy as np
import sklearn 
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.mlab as mlab
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import pickle
import bisect
from scipy.optimize import curve_fit
import scipy.stats as stats



class DataArrangement:

    
    def __init__(self, intensity_data_path, darkness_data_path, baseline_data_path, taker):
        
        self.intensity_df = pd.read_excel(intensity_data_path, engine='openpyxl').rename(columns={'Unnamed: 0': 'time'})
        self.darkness_df = pd.read_excel(darkness_data_path, engine='openpyxl',header=None,names=['wavelength', 'darkness'])
        self.baseline_df = pd.read_excel(baseline_data_path, engine='openpyxl',header=None,names=['wavelength', 'baseline'])
        self.spectra_df = self.calculate_abs()
        taker.keep_raw_data(self.spectra_df)

    def calculate_abs(self):
        df = self.intensity_df.copy()
        d_array = np.array(self.darkness_df['darkness'])
        b_array = np.array(self.baseline_df['baseline'])
        rows = df.shape[0]
        for i in tqdm(range(rows)):
            intensity = np.array(df.iloc[i,1:])#timeを除くため１から
            absorb = self.calculate_A(d_array, b_array, intensity)
            df.iloc[i,1:] = absorb
        df = self.NaN_remove(df)

        return df

    def calculate_A(self, darkness:np.array, baseline:np.array, absorb):
        A = -np.log10((absorb - darkness) / (baseline - darkness))
        return A   
    
    def NaN_remove(self, df):
        df = df.dropna(how='any', axis=1)
        return df

    @staticmethod
    def time_picker(df, times):
        return df[df.eval('{} <= time <= {}'.format(*times))]

    
class NoiseRemover(DataArrangement):
    
    
    def __init__(self, intensity_data_path, darkness_data_path, baseline_data_path, taker):
        super().__init__(intensity_data_path, darkness_data_path, baseline_data_path, taker)
        self.wave_moving_average_df = pd.DataFrame()
        self.time_moving_average_df = pd.DataFrame()
        self.w_t_moving_average_df = pd.DataFrame()
        self.w_window = None
        self.t_window = None
        self.w_t_window = (None, None)
        self.taker = taker
    
    def wave_rolling(self, window, df=pd.DataFrame()):
        if df.empty == True: 
            if self.w_window != window:
                self.w_window = window
            length = len(self.spectra_df)
            frame = pd.DataFrame(columns=list(self.spectra_df.columns))

            for i in tqdm(range(length)):
                series = self.rolling(self.spectra_df.iloc[i,1:], window, centering=True)
                series = series.append(pd.Series(self.spectra_df['time'][i], index=['time']))
                frame = frame.append(series, ignore_index=True)

            self.wave_moving_average_df = frame.reset_index(drop=True).dropna(how='any', axis=1)
            self.taker.keep_arranged_data(self.wave_moving_average_df, 'wave_ave', window)
        else:
            length = len(df)
            frame = pd.DataFrame(columns=list(df.columns))

            for i in tqdm(range(length)):
                series = self.rolling(
                    df.iloc[i,1:],
                    window,
                    centering=True
                    ).append(pd.Series(df['time'][i], index=['time']))
                frame = frame.append(series, ignore_index=True)
            return frame.reset_index(drop=True).dropna(how='any', axis=1)
    
    def time_rolling(self , window, df=pd.DataFrame()):
        if df.empty == True:
            if self.t_window != window:
                self.t_window = window
            width = self.spectra_df.shape[1]
            frame = self.spectra_df.iloc[:,0]
            for i in tqdm(range(width)):
                if i == 0:
                    continue
                else:
                    series = self.rolling(self.spectra_df.iloc[:,i], window, centering=False)
                    frame = pd.concat([frame,series],axis=1)
            self.time_moving_average_df = frame.dropna(how='any', axis=0).reset_index(drop=True)
            self.taker.keep_arranged_data(
                self.time_moving_average_df, 
                'time_ave', 
                window
                )
        else:
            width = df.shape[1]
            frame = df.iloc[:,0]
            for i in tqdm(range(width)):
                if i == 0:
                    continue
                else:
                    series = self.rolling(df.iloc[:,i], window, centering=False)
                    frame = pd.concat([frame,series],axis=1)
            return frame.dropna(how='any', axis=0).reset_index(drop=True)
            
    def rolling(self, x , window, centering):
        return x.rolling(window, center=centering).mean()
    
    def time_wave_average(self, w_window, t_window):
        self.w_t_window = (w_window, t_window)
        df = self.spectra_df
        self.w_t_moving_average_df = self.time_rolling(
            df=self.wave_rolling(df=df, window=w_window),
            window=t_window
            )
        self.taker.keep_arranged_data(
            self.w_t_moving_average_df,
            'wave_time_ave',
            (w_window, t_window)
            )

    def execute_rolling(self, data_name, window):
        if data_name == 'time_ave':
            self.time_roling(window)
        elif data_name == 'wave_ave':
            self.wave_rolling(window)
        elif data_name == 'wave_time_ave':
            self.time_wave_average(*window)

    @staticmethod
    def amperometry_remove(df, f_sec, s_sec):
        first = df.iloc[f_sec[0]:f_sec[1],1:].mean()
        second = df.iloc[s_sec[0]:s_sec[1],1:].mean()
        return pd.concat([first,second], join='inner',axis=1).T


class Displayer:
    
    def __init__(self):
        self.fig_wave = None
        self.fig_time = None
        self.fig_w_t = None

    def display_abs_for_wave(self, df):
        
        length = len(df)
        waves = df.shape[1]
        counter = 0
        color_list = [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf'
        ]
        
        cont = True
        time_list = []
        print('{}~{}で端を指定してください'.format(1,waves - 1))
        left = int(input('左端：'))
        right = int(input('右端：'))
        print('上下の表示範囲を指定してください')
        above = float(input('上：'))
        below = float(input('下：'))
        self.fig_wave = plt.figure(figsize=(12,10))
        while cont:
            
            mess = '{}~{}の値を入力してください。:'.format(0,length - 1)
            time = int(input(mess))
            real_time = df.iloc[time,0]
            time_list.append(real_time)
            print('time:', real_time)
            wave = np.array(df.columns[left:right])
            A = np.array(df.iloc[time, left:right])

            plt.plot(wave, A, lw=0.5, color=color_list[counter],label=str(real_time))

            while True:
                
                q = input('別のデータを追加しますか？[yes,no]:')
                if q == 'yes':
                    break
                elif q == 'no':
                    cont = False
                    break
                else:
                    print('yesかnoで答えろ')
                    continue
            
            counter += 1
            if counter > 10:
                print('これ以上は描画できません。')
                break
                
        print(time_list)
        x_lim = (df.columns[left], df.columns[right])
        y_lim = (below, above)
        plt.grid(True)
        plt.title('Input Wave')
        plt.xlabel('wavelength[nm]')
        plt.ylabel('absorb')
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.legend()
        
    def display_abs_for_time(self ,df):

        length = df.shape[1] - 1
        mess = '{}~{}の値を入力してください。:'.format(1,length)

        wave_index = int(input(mess))
        print(df.columns[wave_index])
        time = np.array(df.iloc[:,0])
        absorb = np.array(df.iloc[:,wave_index])

        self.fig_time = plt.figure(figsize=(12,10))
        plt.plot(time, absorb, color='green', lw=0.5)
        plt.scatter(time, absorb, c='green', lw=0.3)
        plt.grid(True)
        plt.title('Input Wave')
        plt.xlabel('time[ms]')
        plt.ylabel('absorb')
        
    def display_3D(self, df, el, az):

        time_length = df.shape[0]
        wave_num = df.shape[1]
        print('{}〜{}で'.format(0,wave_num - 1),end='')
        mess = '左端を指定してください：'
        left_num = int(input(mess))
        mess = '右端を指定してください：'
        right_num = int(input(mess))

        test_data = np.array(df.iloc[0:time_length,left_num:right_num])

        x = np.array(list(df.columns[left_num:right_num]))
        y = np.array(list(df['time']))
        self.three_dimention_display(x, y, test_data, el, az)

    def three_dimention_display(self, x, y, z, el, az):
        X, Y = np.meshgrid(x, y)
        self.fig_w_t = plt.figure(figsize=(12, 10),facecolor="w")
        ax = Axes3D(self.fig_w_t)
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel("wave_length[nm]", size = 14)
        ax.set_ylabel("time[ms]", size = 14)
        ax.set_zlabel("absorbance", size = 14)
        surface = ax.plot_surface(X, Y, z,cmap=cm.gist_earth,linewidth=0, antialiased=False,alpha=1) 
        self.fig_w_t.colorbar(surface, ax=ax, shrink=0.5)  

    def display_model_and_data(self, df, model, wave):

        time = np.array(df.iloc[:,0])
        absorb = np.array(df[wave])
        self.fig_time = plt.figure(figsize=(12,10))
        plt.plot(time, absorb, color='green', lw=0.5)
        plt.scatter(time, absorb, c='green', lw=0.3)
        predict = model.exponential_func(time)
        plt.plot(time, predict,color='red', lw=0.7)
        plt.grid(True)
        plt.title('Input Wave')
        plt.xlabel('time[s]')
        plt.ylabel('absorb')

    def display_distribution(self, array, model):
        n, bins, patches = plt.hist(array, bins=int(len(array) / 5), alpha=0.5, normed=1)
        model.gaussian_fitting(array)
        y = stats.norm.pdf(bins, *model.gauss_params)
        pdf = stats.norm.pdf(bins)
        l = plt.plot(bins, y, 'r--',bins, pdf, linewidth=2)
        plt.grid(True)
        plt.title('$Histgram\ of \ Noise\  \mu={},\ \sigma={}$'.format(*model.gauss_params))
        plt.show()
        stats.probplot(array, dist="norm", plot=plt)
        plt.show()
        print(stats.shapiro(array))


class Analyser:
    
    def __init__(self, intensity_path=None, d_path=None, b_path=None):
        self.intense_path = intensity_path
        self.d_path = d_path
        self.b_path = b_path
        self.taker = DataTaker()
        self.remover = NoiseRemover(intensity_path, d_path, b_path, self.taker)
        self.drawer = Displayer()
        self.save_state = None
        if intensity_path == None or d_path == None or b_path == None:
            print('実行不可能です。dataをロードするか、パスをAnalyserに与えてください。')
        
    def display(self, data, displ):

        if data == 'time_ave':
            w = int(input('time_window size:'))
            df = self.taker.data_take(data, w)
            if df is None:
                self.remover.execute_rolling(data, w)
                df = self.taker.data_take(data, w)  
            self.assign_display(df, displ)
            self.save_state = displ

        elif data == 'wave_ave':
            w = int(input('window size:'))
            df = self.taker.data_take(data, w)
            if df is None:
                self.remover.execute_rolling(data, w)
                df = self.taker.data_take(data, w)  
            self.assign_display(df, displ)
            self.save_state = displ

        elif data == 'wave_time_ave':
            w = int(input('波長のwindow size:'))
            t = int(input('時間ののwindow size:'))   
            w_t = (w, t)
            df = self.taker.data_take(data, w_t)
            if df is None:
                self.remover.execute_rolling(data, w_t)
                df = self.taker.data_take(data, w_t)  
            self.assign_display(df, displ)    
            self.save_state = displ

        elif data == 'raw':
            self.assign_display(self.remover.spectra_df, displ)
            self.save_state = displ

        else:
            print('time_ave or wave_ave or wave_time_ave or raw のいづれかを入力してください。')
        
    def assign_display(self, df, display):
        if display == '3D':
            print('アングルを指定してください')
            elev = int(input('elev:'))
            azim = int(input('axim:'))
            self.drawer.display_3D(df ,el=elev, az=azim) 
        elif display == 'time':
            self.drawer.display_abs_for_time(df)
        elif display == 'wave':
            self.drawer.display_abs_for_wave(df)
        else:
            print('time か wave か 3D を入力してください')

    def save_fig(self):
        if self.save_state == 'time':
            name = str(input('ファイル名を指定してください：'))
            self.drawer.fig_time.savefig(name + '.png')
        elif self.save_state == 'wave':
            name = str(input('ファイル名を指定してください：'))
            self.drawer.fig_wave.savefig(name + '.png')
        elif self.save_state == '3D':
            name = str(input('ファイル名を指定してください：'))
            self.drawer.fig_w_t.savefig(name + '.png')
    
    def cal_rrc(self, data_name, window, quantile):
        '''
        w = int(input('wave_window size:'))
        if self.remover.w_window != w:
            self.remover.wave_rolling(window=w)
            '''
        df = self.taker.data_take(data_name, window)
        if df is None:
            self.remover.execute_rolling(data_name, window)
            df = self.taker.data_take(data_name, window)
        wave_list = list(df.columns[1:])
        time_list = list(df['time'])
        print('使用する波長を{}~{}で選択してください'.format(wave_list[0], wave_list[-1]))
        wl = int(input('波長:'))
        selected_wave = self.wave_picker(wave_list, wl)
        trigger_time = int(input('トリガーを掛けた時間：'))
        before_trigger = (trigger_time -1000, trigger_time - 0.001)
        learning_time = int(input('triggerを掛けてからどこまでの時間を近似に使いますか？'))
        after_trigger = (trigger_time, time_list[-1])
        learn_span = (trigger_time, learning_time)
        end_time = (time_list[-1] - 1000, time_list[-1])
        Gauss = ModelFitting()

        Gauss.gaussian_fitting(
            np.array(
                self.taker.remove_outliers(
                    self.taker.wave_for_times(
                        df[['time', selected_wave]], before_trigger
                        ),
                    selected_wave,
                    quantile
                    )[selected_wave]
                ).flatten()
        )
        const_AB = Gauss.gauss_params[0]
        print('AB:',const_AB)
        Gauss.gaussian_fitting(
            np.array(
                self.taker.remove_outliers(
                    self.taker.wave_for_times(
                        df[['time', selected_wave]], end_time
                        ),
                    selected_wave,
                    quantile
                    )[selected_wave]
                ).flatten()
        )
        const_B = Gauss.gauss_params[0]
        print('B:',const_B)
        print('A:',const_AB - const_B)

        df_for_draw = self.taker.time_shift(
            DataArrangement.time_picker(
                df[['time', selected_wave]], 
                after_trigger
            )
        )

        df_for_learn = self.taker.time_shift(
            DataArrangement.time_picker(
                df[['time', selected_wave]],
                learn_span
            )
        )

        fitter = ModelFitting()
        fitter.exponential_fitting(df_for_learn, const_AB, const_B)

        return fitter, selected_wave, df_for_draw
    
    def model_display(self, data_name='wave_ave', window=1, model_name='nonlinear', quantile=0.05):
        if model_name == 'nonlinear':
            self.save_state = 'time'
            model, wave , df= self.cal_rrc(data_name, window, quantile)
            self.drawer.display_model_and_data(df, model, wave)
        elif model_name == 'gradient':
            pass

    def draw_absorb_distribution(self, data_name, window=1, quantile=0.01):
        df = self.taker.data_take(data_name, window)
        if df is None:
            self.remover.execute_rolling(data_name, window)
            df = self.taker.data_take(data_name, window)
        #df = self.remover.wave_moving_average_df
        wave_list = list(df.columns[1:])
        time_list = list(df['time'])
        print('使用する波長を{}~{}で選択してください'.format(wave_list[0], wave_list[-1]))
        wl = int(input('波長:'))
        print('使用する時刻範囲を{}~{}で選択してください'.format(time_list[0], time_list[-1]))
        l_time = int(input('左端:'))
        r_time = int(input('右端:'))
        time_span = (
            self.taker.time_picker(time_list, l_time), 
            self.taker.time_picker(time_list, r_time)
            )

        data, wave = self.taker.time_for_one_wave(df, wl)
        print('time_span:', time_span, '波長:', wave)
        wave_for_times = pd.DataFrame(self.taker.wave_for_times(data, time_span).iloc[:,1])
        print(wave_for_times)
        Gauss = ModelFitting()
        self.drawer.display_distribution(
            np.array(self.taker.remove_outliers(
                wave_for_times, 
                wave, 
                quantile)).flatten(), 
            Gauss
            )
        
    def wave_picker(self, wave_list, wave):
        return wave_list[bisect.bisect(wave_list, wave)]

    def save_instance(self):
        Saver().save_data(self)

    @staticmethod
    def load_instance():
        loader = Saver().load_data()

        return loader


class DataTaker:

    def __init__(self):
        self.wave_data = {}
        self.time_data = {}
        self.wave_time_data = {}
        self.spectra_data = None

    def keep_raw_data(self, spectra_df):
        self.spectra_data = spectra_df

    def keep_arranged_data(self, data, data_name, window_size):
        if data_name == 'time_ave':
            self.time_data[window_size] = data
        elif data_name == 'wave_ave':
            self.wave_data[window_size] = data
        elif data_name == 'wave_time_ave':
            self.wave_time_data[window_size] = data

    def data_take(self, data_name, window=1):
        if data_name == 'raw':
            return self.spectra_data
        elif data_name == 'time_ave' and window in tuple(self.time_data.keys()):
            return self.time_data[window]
        elif data_name == 'wave_ave' and window in tuple(self.wave_data.keys()):
            return self.wave_data[window]
        elif data_name == 'wave_time_ave' and window in tuple(self.wave_time_data.keys()):
            return self.wave_time_data[window]
        else:
            return None

    def time_for_one_wave(self, df, wave):
        wave_list = list(df.columns[1:])
        wave = self.wave_picker(wave_list, wave)
        return df[['time', wave]] , wave

    def wave_for_times(self, df, time_span):
        return df[df.eval('{} <= time <= {}'.format(*time_span))]

    def wave_picker(self, wave_list, wave):
        return wave_list[bisect.bisect(wave_list, wave)]

    def time_picker(self, time_list, time):
        return time_list[bisect.bisect(time_list, time)]
    
    def remove_outliers(self, df, column, q):
        q_val_left = df[column].quantile(q)
        q_val_right = df[column].quantile(1 - q)
        return df[(df[column] > q_val_left) & (df[column] < q_val_right)]

    def time_shift(self, df):
        df['time'] = df['time'].map(lambda x: (x -df['time'].min()) / 1000) 
        return df
            

class ModelFitting:

    def __init__(self):
        self.exp_param = None
        self.gauss_params = ()

    def exponential_fitting(self, df, const_AB, const_B):
        const_A = const_AB - const_B
        self.const_A = const_A
        self.const_B = const_B
        def exponential(x, k):
            y_hat = const_A * (np.exp(-k*x)) + const_B
            return y_hat

        x_name = df.columns[0]
        y_name = df.columns[1]
        x = np.array(df[x_name])
        y = np.array(df[y_name])
        popt, pcov = curve_fit(exponential, x, y)
        print('k:',popt)
        self.exp_param = popt
    
    def exponential_func(self, x):
        return self.const_A * (np.exp(-self.exp_param*x)) + self.const_B

    def gaussian_fitting(self, x):
        (mu, sigma) = stats.norm.fit(x)
        self.gauss_params = (mu, sigma)


class Saver:

    def save_data(self, data):
        file_name = input('ファイル名を入力してください：')
        file_name = file_name + '.pkl'        
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self):
        file_name = input('読み込むファイル名：')
        with open(file_name, 'rb') as f:
            load = pickle.load(f)
        
        return load



