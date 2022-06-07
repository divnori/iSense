# Use this for debugging to reload 
# import sys; reload(sys.modules['metrics']); from metrics import BinaryClassificationMetrics

from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from numbers import Number
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


class BinaryClassificationMetrics(object):
    '''
    num_thresh - number of thresholds equally spaced in [0,1] at which the confusion matrix is computed
    '''
    def __init__(self, num_thresh=101):
        self._numthresh = num_thresh
        self._modname,self._modname_dct = list(),dict()
        self._modname_sz = list()
        self._scores = list()
        self._confmat = list()
        self._auc,self._prrec = list(),list()


    '''
    name - model name
    scores_pd - pandas dataframe with columns named label and score
        label: true labels with ones (events) and zeros (non-events)
        score: model output; scores in [0,1]
    params - optional dictionary of parameters
        skl_auc_average: micro, macro, weighted, samples (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
        skl_ap_average: micro, macro, weighted, samples (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
    '''
    def addModel(self, name, scores_pd, params={}):
        self._modname_dct[name] = len(self._modname)
        self._modname.append(name)
        self._modname_sz.append('%s (%s)'%(name,'{:,}'.format(len(scores_pd))))
        self._scores.append(scores_pd)

        labels,scores = scores_pd['label'].values,scores_pd['score'].values
        thresholds = np.linspace(0,1,self._numthresh)

        #confusion matrix at various thresholds
        tot,pos = len(labels),sum(labels)
        posarr,negarr = labels==1,labels==0
        tp,tn,fp,fn = (map(lambda x:np.zeros(self._numthresh), range(4)))
        for i,t in enumerate(thresholds):
            tmp = scores>=t
            tp[i] = np.logical_and(tmp,posarr).sum()
            fp[i] = np.logical_and(tmp,negarr).sum()
            fn[i],tn[i] = pos-tp[i],tot-pos-fp[i]

        with np.errstate(divide='ignore', invalid='ignore'):
            sens,spec,ppv,npv,accu,prev = tp/(tp+fn), tn/(tn+fp), tp/(tp+fp), tn/(tn+fn), (tp+tn)/tot, (tp+fn)/tot
            lift,f1 = sens/prev,2/(1/ppv + 1/sens)

        df = pd.DataFrame(OrderedDict([
            ('thresh',thresholds),('tp',tp.astype(int)),('tn',tn.astype(int)),('fp',fp.astype(int)),('fn',fn.astype(int)),
                ('sens',sens),('spec',spec),('ppv',ppv),('npv',npv),('accu',accu),('prev',prev),('lift',lift),('f1',f1)   
        ]))
        self._confmat.append(df)

        #ROC curve - area
        self._auc.append(roc_auc_score(labels,scores,params.get('skl_auc_average','micro')))
 
        #precision recall curve - average precision
        self._prrec.append(average_precision_score(labels,scores,params.get('skl_ap_average','micro')))


    '''
    model_names: list of model names
    return:    if model_names is empty, indices for all models are returned
               otherwise, each name is checked against the model names currently in the object and their indices are returned
    '''
    def getModelIndexes(self, model_names=[]):
        return range(len(self._modname)) if model_names is None or len(model_names)==0 else \
            filter(lambda x:x is not None, map(lambda x:self._modname_dct.get(x), model_names))


    '''
    model_names:   list of model names to be plotted
    chart_types: 1=ScoreDistribtion or 2=ConfusionMatrix for different thresholds
    params - parameters used to create plots
        legloc: location of the legend (1=TR, 2=TL, 3=BL, 4=BR), can also be x,y coordinates eg (.5,.05)
    '''
    def plot(self, model_names=[], chart_types=[], params={}):
        model_idx = self.getModelIndexes(model_names)
        chart_types = [1,2,3] if chart_types is None or len(chart_types)==0 else list(filter(lambda x:x in [1,2,3], chart_types))
        save,pfx = params.get('save',False),params.get('prefix','')
        fs_ti,fs_ax,fs_le,fs_tk = 17,15,14,12

        def ShowOrSave(s):
            if save:
                plt.savefig(s, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

        def ScoreDistribution(inp_pd, mname):
            labels,scores = inp_pd['label'].values,inp_pd['score'].values
            eve,nev = scores[labels==1],scores[labels==0]
            n1,m1,s1 = len(eve),np.mean(eve),np.std(eve)
            n0,m0,s0 = len(nev),np.mean(nev),np.std(nev)

            bins = np.linspace(0,1,100)
            plt.figure(figsize=(14,6))
            plt.hist(eve, bins, alpha=.6, density=True, color='navy', label='Events     %s ($\mu$=%.2f, $\sigma$=%.2f)'%(format(n1,','),m1,s1))
            plt.hist(nev, bins, alpha=.6, density=True, color='darkorange', label='Non-events %s ($\mu$=%.2f, $\sigma$=%.2f)'%(format(n0,','),m0,s0))
            plt.xlim([-0.01, 1.01])
            plt.xlabel('Score Bin', fontsize=fs_ax)
            plt.ylabel('Percentage of Observations Per Class', fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight='bold')
            plt.legend(loc=params.get('legloc',1), prop={'size':fs_le,'family':'monospace'})
            plt.tick_params(axis='both', which='major', labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,pos:'%3d%%'%x))
            ShowOrSave('%s%s-scores.png'%(pfx,mname))

        def TPFPTNFN(cmat_pd, mname):
            sz = .01*(cmat_pd['tp'][0]+cmat_pd['tn'][0]+cmat_pd['fp'][0]+cmat_pd['fn'][0])
            thresh,tp,tn,fp,fn = cmat_pd['thresh'].values,cmat_pd['tp'].values/sz,cmat_pd['tn'].values/sz,cmat_pd['fp'].values/sz,cmat_pd['fn'].values/sz

            plt.figure(figsize=(14,6))
            plt.plot(thresh, tp, color='navy', label='TP')
            plt.plot(thresh, fp, color='navy', label='FP', linestyle='--')
            plt.plot(thresh, tn, color='darkorange', label='TN')
            plt.plot(thresh, fn, color='darkorange', label='FN', linestyle='--')
            plt.xlim([-0.01, 1.01])
            plt.xlabel('Thresholds', fontsize=fs_ax)
            plt.ylabel('Percentage of All Observations', fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight='bold')
            plt.legend(loc=params.get('legloc',4), prop={'size':fs_le,'family':'monospace'})
            plt.tick_params(axis='both', which='major', labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,pos:'%3d%%'%x))
            ShowOrSave('%s%s-cmat.png'%(pfx,mname))

        def Accuracy(cmat_pd, mname):
            thresh,sens,spec,accu = cmat_pd['thresh'].values,100.*cmat_pd['sens'].values,100.*cmat_pd['spec'].values,100.*cmat_pd['accu'].values        

            plt.figure(figsize=(14,6))
            plt.xlim([-0.01, 1.01])
            plt.plot(thresh, accu, color='black', label='accuracy')
            idx =  np.nanargmax(accu)
            plt.plot(thresh[idx], accu[idx], 'x', color='black', markersize=10, zorder=200, label='(%.2f,%.1f%%)'%(thresh[idx],accu[idx]))
            plt.plot(thresh, sens, color='blue', label='sensitivity')
            plt.plot(thresh, spec, color='red', label='specificity')
            idx =  np.nanargmin(abs(sens-spec))
            plt.plot(thresh[idx], sens[idx], 'o', color='magenta', markerfacecolor='none', markersize=10, zorder=100, label='(%.2f,%.1f%%)'%(thresh[idx],accu[idx]))
            plt.xlabel('Thresholds', fontsize=fs_ax)
            plt.ylabel('Percentage of Observations', fontsize=fs_ax)
            plt.title(mname, fontsize=fs_ti, fontweight='bold')
            plt.legend(loc=params.get('legloc',4), prop={'size':fs_le,'family':'monospace'})
            plt.tick_params(axis='both', which='major', labelsize=fs_tk)
            plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,pos:'%3d%%'%x))
            ShowOrSave('%s%s-accu.png'%(pfx,mname))

        for idx in list(model_idx):
            if 1 in chart_types:
                ScoreDistribution(self._scores[idx], self._modname[idx])
            if 2 in chart_types:
                TPFPTNFN(self._confmat[idx], self._modname[idx])
            if 3 in chart_types:
                Accuracy(self._confmat[idx], self._modname[idx])


    '''
    model_names: list of model names to be plotted
    chart_types: 1=Receiver Operating Characteristics (ROC) or 2=Precision Recall for different thresholds
    params - parameter used to create plots
        legloc: location of the legend (1=TR, 2=TL, 3=BL, 4=BR), can also be x,y coordinates eg (.5,.05)
        save:   boolean, save chart to disk
        pfx:    prefix to filename if saved to disk, used only when save=True
        addsz:  boolean, add number of observations used to compute the AUC/AP
    '''
    def plotROC(self, model_names=[], chart_types=[], params={}):
        model_idx = self.getModelIndexes(model_names)
        chart_types = [1,2] if chart_types is None or len(chart_types)==0 else list(filter(lambda x:x in [1,2], chart_types))
        save,pfx = params.get('save',False),params.get('prefix','')
        names = self._modname_sz if params.get('addsz',True) else self._modname
        plotthresh = params.get('showthresh',[])

        def ROCPR(midx, ctype, labs):
            plt.figure(figsize=(8,8))
            for m in midx:
                thresh,spec,sens,ppv = self._confmat[m][['thresh','spec','sens','ppv']].values.transpose()
                if ctype==1:
                    p = plt.plot(1-spec, sens, label='%s %0.4f' %(names[m],self._auc[m]))
                else:
                    p = plt.plot(sens, ppv, label='%s %0.4f' %(names[m],self._prrec[m]))

                for th in plotthresh:
                    idx = np.argmin(abs(thresh-th))
                    if ctype==1:
                        plt.plot(1-spec[idx], sens[idx], 'o', color=p[0].get_color(), markersize=6, zorder=200)
                    else:
                        plt.plot(sens[idx], ppv[idx], 'o', color=p[0].get_color(), markersize=6, zorder=200)

            if ctype==1:
                plt.plot([0, 1], [0, 1], color='black', linestyle=':')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel(labs[0], fontsize=15)
            plt.ylabel(labs[1], fontsize=15)
            plt.legend(loc=params.get('legloc',1), prop={'size':14,'family':'monospace'})
            plt.tick_params(axis='both', which='major', labelsize=12)

            if save:
                lbl = 'roc' if ctype==1 else 'pr'
                plt.savefig('%s%s.png'%(pfx,lbl), dpi=150, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

        if 1 in chart_types:
            labs = ('False Positive Rate (1-Specificity)','True Positive Rate (Sensitivity)')
            ROCPR(model_idx, 1, labs)
        if 2 in chart_types:
            labs = ('Recall (Sensitivity)','Precision (Positive Predictive Value)')
            ROCPR(model_idx, 2, labs)


    '''
    model_names: list of models for which thresholds are computed
    fpwt, fnwt: weight applied on false positives and false negatives
    Return the confusion matrix for which fpwt x #FP = fnwt x #FN
    '''
    def confusionMatrixWeights(self, model_names=[], fpwt=1, fnwt=1):
        model_idx = self.getModelIndexes(model_names)

        out = pd.DataFrame()
        for m in model_idx:
            idx = np.argmin(abs(fpwt*self._confmat[m]['fp'].values-fnwt*self._confmat[m]['fn'].values))
            out = out.append(self._confmat[m].iloc[[idx]], ignore_index=True)
        out.insert(0, 'model', map(lambda x:self._modname[x], model_idx))

        return out


    '''
    model_names: list of models for which thresholds are computed
    key:        'thresh', 'sens', 'spec', 'ppv', 'npv'
    value:      floating point number; if this is empy, the confusion matrix corresponding to max value of this param is returned
    Return the confusion matrix which matches value in a key
    '''
    def confusionMatrixKeyValue(self, model_names=[], key='thresh', value=None):
        assert key in self._confmat[0].columns, 'Error: Key not found in confustion matrix dataframe'
        model_idx = self.getModelIndexes(model_names)
        flag = True if isinstance(value,Number) else False

        out = pd.DataFrame()
        for m in model_idx:
            idx =  np.nanargmin(abs(self._confmat[m][key].values-value)) if flag else np.nanargmax(abs(self._confmat[m][key].values))
            out = out.append(self._confmat[m].iloc[[idx]], ignore_index=True)
        out.insert(0, 'model', list(map(lambda x:self._modname[x], model_idx)))

        return out

