#Import package 
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
pal = sns.color_palette()

from sklearn.feature_selection import chi2
from scipy.stats import chi2_contingency
import streamlit as st
from io import StringIO
import sys
from io import BytesIO
from PIL import Image

#---------------------Tampilan informasi
st.sidebar.write("""## GUI - python untuk pemodelan machine learning""")
st.sidebar.write("""##### _Created by: Riza Fahlevi_""")

#-----Tampilan pada sidebar
option = st.sidebar.selectbox(
    'Pilih Tampilan Layar:',
    ('Home', 'Output')
)
st.sidebar.write("""###### note: silahkan pilih 'Olah Data' untuk memulai olah data""")

#-----Read Data
st.sidebar.write("#### 1. Upload Data")
file = st.sidebar.file_uploader('Upload Data', type=['xlsx'])

#-----Tampilan sidebar untuk metode korelasi
st.sidebar.write("#### 2. Correlation Matrix")
option1 = st.sidebar.selectbox(
    'Pilih Metode:',
    ('','Kendall', 'Pearson', 'Spearman')
)
#-----Tampilan sidebar untuk Uji pengaruh
st.sidebar.write("#### 3. Uji Signifikansi")
option2 = st.sidebar.selectbox(
    'Pilih Metode:',
    ('','Uji T', 'Chi-Square')
)
#-----Tampilan sidebar untuk Memilih model
st.sidebar.write("#### 4. Pilih Model")
option3 = st.sidebar.selectbox(
    'Pilih Metode:',
    ('','Decission Tree', 'Adaboost', 'KNN', 'XGBoost', 'Random Forest', 'CATBoost')
)

#Tampilan slider train and test split
st.sidebar.write("##### Train & Test Split Data")
split = st.sidebar.slider('Choose your test size', 0.0, 1.0, 0.2)

#-----Tampilan untuk process
st.sidebar.write("#### 5. Process Data")
prepros = st.sidebar.button('Process')
                            
    
#---------------------Tampilan dan urutan mengeluarkan output 
if option == 'Home' or option == '':
    st.markdown("""
        <h2 style='text-align: center;'>Implementasi Graphical User Interface (GUI) - Python dalam pemodelen machine learning untuk analisa pengaruh antar variabel</h2>
        <h4 style='text-align: center;'>Studi kasus: Customer profile terhadap RAI pada product NDF Car</h4>
    """, unsafe_allow_html=True)
    
    image = Image.open('BFI-Finance.png')
    st.image(image)
    
elif option == 'Output':
    st.header('Output Olah Data:')
    
    if file is not None:
        #read data
        data = pd.read_excel(file)
        
        # Tampilkan dataframe
        st.write(" Data Review:")
        st.dataframe(data)
        
        def capture_output():
            #Menggunakan StringIO untuk menangkap output
            output = StringIO()
            sys.stdout = output
            return output

        # Mengembalikan output ke sys.stdout
        def release_output(output):
            sys.stdout = sys.__stdout__
            return output.getvalue()

        # Menangkap output model summary
        output = capture_output()
        data.info()
        summary_output = release_output(output)

        # Menampilkan Model Summary dengan CNN
        st.write('Data Summary:')
        st.code(summary_output, language='plaintext')

    #Tombol preprocessing
    if prepros:  
        if option1 == 'Kendall':
            plt.figure(figsize = (18, 7))
            sns.heatmap(data.corr(method='kendall'), annot = True, fmt = '0.2f', annot_kws = {'size' : 15}, linewidth = 5, linecolor = 'skyblue', cmap = 'Blues')
            st.write('\n')
            st.header('**Correlation Matrix (Kendall):**')
            st.pyplot(plt)
            
        elif option1 == 'Pearson':
            plt.figure(figsize = (18, 7))
            sns.heatmap(data.corr(method='pearson'), annot = True, fmt = '0.2f', annot_kws = {'size' : 15}, linewidth = 5, linecolor = 'skyblue', cmap = 'Blues')
            st.write('\n')
            st.header('**Correlation Matrix:**')
            st.pyplot(plt)
            
        elif option1 == 'Spearman':
            plt.figure(figsize = (18, 7))
            sns.heatmap(data.corr(method='spearman'), annot = True, fmt = '0.2f', annot_kws = {'size' : 15}, linewidth = 5, linecolor = 'skyblue', cmap = 'Blues')
            st.write('\n')
            st.header('**Correlation Matrix:**')
            st.pyplot(plt)
            
        else:    
            st.write('Pilih opsi correlation matrix terlebih dahulu')
        
        if option2 == 'Uji T':
            st.write('belum ada sintaks')
        
        elif option2 == 'Chi-Square':
            #Pendefinisian Model
            def chi2_GUI(col1, col2, f, f1):    
                df_crosstab= pd.crosstab(index = col1, columns = col2)
                degree_f = (df_crosstab.shape[0]-1) * (df_crosstab.shape[1]-1)
                df_crosstab.loc[:,'Total']= df_crosstab.sum(axis=1)
                df_crosstab.loc['Total']= df_crosstab.sum()
    
                df_exp = df_crosstab.copy()    
                df_exp.iloc[:,:] = np.multiply.outer(
                    df_crosstab.sum(1).values,df_crosstab.sum().values) / df_crosstab.sum().sum()            
            
                df_chi2 = ((df_crosstab - df_exp)**2) / df_exp    
                df_chi2.loc[:,'Total']= df_chi2.sum(axis=1)
                df_chi2.loc['Total']= df_chi2.sum()
    
                chi_square_score = df_chi2.iloc[:-1,:-1].sum().sum()
    
                from scipy import stats
                from scipy.stats import chi2
                alpha=0.05
                p = stats.distributions.chi2.sf(chi_square_score, degree_f)
                critical_value=chi2.ppf(q=1-alpha,df=degree_f)
    
                st.write('chi_square:',chi_square_score)
                st.write('critical_value:',critical_value)
                st.write('Df:', degree_f)
                st.write('p-value:',p)
                st.write('alpha:',alpha)
                st.write('\n')
                st.write('**Kesimpulan**')
                if p<=alpha:
                    st.write('Reject H0, There is a relationship between variable', f, 'and', f1,'(Target Variable)')
                else:
                    st.write('Retain H0, There is no relationship between variable', f, 'and', f1,'(Target Variable)') 
                return
            
            f=data.columns.tolist()
            
            st.header('**UJI SIGNIFIKANSI**')
            st.write('____________________')
            
            #looping uji pengaruh antar variabel (chi2)
            n=0
            while n >= 0:
                n+=1
                try:
                    st.write(f[n], 'to',f[0])
                    st.write(chi2_GUI(data.iloc[:,0],data.iloc[:,n],f[n],f[0]))
                    st.write('_____________')
                except IndexError:
                    break
        else:    
            st.write('Pilih opsi uji pengaruh terlebih dahulu')
            
        # pemisahan features vs target
        X = data.drop(data.columns[0],axis=1)
        y = data[data.columns[0]].values
    
        # Mendefinisikan variabel
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        from sklearn.model_selection import cross_validate
        from sklearn.utils import compute_sample_weight

        def eval_classification(model):
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            y_pred_proba = model.predict_proba(X_test)
            y_pred_proba_train = model.predict_proba(X_train)

            st.write("Accuracy (Test Set): %.2f" % accuracy_score(y_test, y_pred))
            st.write("Precision (Test Set): %.2f" % precision_score(y_test, y_pred))
            st.write("Recall (Test Set): %.2f" % recall_score(y_test, y_pred))
            st.write("F1-Score (Test Set): %.2f" % f1_score(y_test, y_pred))
            st.write("")
            st.write("roc_auc (train-proba): %.2f" % roc_auc_score(y_train, y_pred_proba_train[:, 1]))
            st.write("roc_auc (test-proba): %.2f" % roc_auc_score(y_test, y_pred_proba[:, 1]))

        def eval_cv_ab_roc_auc(model):
            score = cross_validate(model, X_train, y_train, cv=5, scoring='roc_auc', return_train_score=True)
            st.write('roc_auc (crossval train): '+ str(score['train_score'].mean()))
            st.write('roc_auc (crossval test): '+ str(score['test_score'].mean()))

        def eval_cv_ab_precision(model):
            score = cross_validate(model, X_train, y_train, cv=5, scoring='precision', return_train_score=True)
            st.write('precision (crossval train): '+ str(score['train_score'].mean()))
            st.write('precision (crossval test): '+ str(score['test_score'].mean()))

        def eval_cv_ab_recall(model):
            score = cross_validate(model, X_train, y_train, cv=5, scoring='recall', return_train_score=True)
            st.write('recall (crossval train): '+ str(score['train_score'].mean()))
            st.write('recall (crossval test): '+ str(score['test_score'].mean()))

        def eval_cv_ab_accuracy(model):
            score = cross_validate(model, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True)
            st.write('accuracy (crossval train): '+ str(score['train_score'].mean()))
            st.write('accuracy (crossval test): '+ str(score['test_score'].mean()))

        def weighted_sample(y):
            class_weights = compute_sample_weight(class_weight={0: 1, 1: 2}, y=y)  # Calculate sample weights based on class imbalance
            return class_weights
        
                
        def show_best_hyperparameter(model):
            st.write(model.best_estimator_.get_params())
            

        if option3 == 'Decission Tree':
            from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
            import numpy as np

            # List of hyperparameter
            max_depth = [int(x) for x in np.linspace(1, 10, num = 10)] # Maximum number of levels in tree
            min_samples_split = [2, 5, 10, 100] # Minimum number of samples required to split a node 2, 5, 10, 100
            min_samples_leaf = [1, 2, 4, 10, 20, 50] # Minimum number of samples required at each leaf node 1, 2, 4, 10, 20, 50
            max_features = ['auto', 'sqrt'] # Number of features to consider at every split
            criterion = ['gini','entropy']
            splitter = ['best','random']

            hyperparameters = dict(max_depth=max_depth, 
                                   min_samples_split=min_samples_split, 
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   criterion=criterion,
                                   splitter=splitter
                                  )

            # Init
            from sklearn.tree import DecisionTreeClassifier
            dt = DecisionTreeClassifier(random_state=42)
            dt_tuned = RandomizedSearchCV(dt, hyperparameters, cv=5, random_state=42, scoring='recall')
            dt_tuned.fit(X_train,y_train)

            # Predict & Evaluation
            st.write(eval_classification(dt_tuned))
            st.write(eval_cv_ab_accuracy(dt_tuned))

            # Show Best Hyperparameter
            # st.write(show_best_hyperparameter(dt_tuned))

            from sklearn.metrics import ConfusionMatrixDisplay          
            # Confusion matrix
            st.write("Result Training")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    dt_tuned,
                    X_train,
                    y_train,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            st.write("Result Test")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    dt_tuned,
                    X_test,
                    y_test,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            # Show feature importance
            st.header("**Feature importance**")
            fig, axes = plt.subplots(ncols=1, figsize=(20,8))
            feat_importances = pd.Series(dt_tuned.best_estimator_.feature_importances_, index=X.columns)
            ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 8))
            ax.invert_yaxis()
            plt.xlabel('score')
            plt.ylabel('feature')
            plt.title('feature importance score')
            st.pyplot(plt)
            
            
        elif option3 == 'Adaboost':   
            from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
            import numpy as np

            # List of hyperparameter
            hyperparameters = dict(n_estimators = [int(x) for x in np.linspace(start = 1, stop = 200, num = 20)], # Jumlah iterasi
                                   learning_rate = [float(x) for x in np.linspace(start = 0.001, stop = 1, num = 20)],  
                                   algorithm = ['SAMME', 'SAMME.R']
                                  )

            # Init model
            from sklearn.ensemble import AdaBoostClassifier
            ab = AdaBoostClassifier(random_state=42)
            ab_tuned = RandomizedSearchCV(ab, hyperparameters, random_state=42, cv=5, scoring='recall')
            ab_tuned.fit(X_train,y_train)

            # Predict & Evaluation
            st.write(eval_classification(ab_tuned))
            st.write(eval_cv_ab_accuracy(ab_tuned))
            #st.write(show_best_hyperparameter(ab_tuned))
            from sklearn.metrics import ConfusionMatrixDisplay
            # Confusion matrix
            st.write("Result Training")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    ab_tuned,
                    X_train,
                    y_train,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            st.write("Result Test")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    ab_tuned,
                    X_test,
                    y_test,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            st.header("**Feature importance**")
            fig, axes = plt.subplots(ncols=1, figsize=(20,8))
            feat_importances = pd.Series(ab_tuned.best_estimator_.feature_importances_, index=X.columns)
            ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 8))
            ax.invert_yaxis()
            plt.xlabel('score')
            plt.ylabel('feature')
            plt.title('feature importance score')
            st.pyplot(plt)   
            
        
        elif option3 == 'KNN':
            from sklearn.model_selection import RandomizedSearchCV

            # List of Hyperparameter
            n_neighbors = list(range(1,110))
            p=list(range(1,20))
            algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
            hyperparameters = dict(n_neighbors=n_neighbors, p=p, algorithm=algorithm)

            # Init Model
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier()
            knn.fit(X_train, y_train)
            knn_tuned = RandomizedSearchCV(knn, hyperparameters, scoring='recall', random_state=1, cv=5)
            knn_tuned.fit(X_train, y_train)

            # Predict & Evaluation
            eval_classification(knn_tuned)
            eval_cv_ab_accuracy(knn_tuned)
            #show_best_hyperparameter(knn_tuned)
            
            from sklearn.metrics import ConfusionMatrixDisplay
            # Confusion matrix
            st.write("Result Training")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    knn_tuned,
                    X_train,
                    y_train,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            st.write("Result Test")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    knn_tuned,
                    X_test,
                    y_test,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
        
        elif option3 == 'XGBoost':
            st.header("**XGBoost**")
            from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
            import numpy as np
            
            #Menjadikan ke dalam bentuk dictionary
            hyperparameters = {
                                'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)],
                                'min_child_weight' : [int(x) for x in np.linspace(1, 20, num = 15)],
                                'gamma' : [float(x) for x in np.linspace(0, 1, num = 11)],
                                'tree_method' : ['auto', 'exact', 'approx', 'hist'],

                                'colsample_bytree' : [float(x) for x in np.linspace(0, 1, num = 11)],
                                'eta' : [float(x) for x in np.linspace(0.01, 0.2, num = 100)],

                                'lambda' : [float(x) for x in np.linspace(0, 1, num = 11)],
                                'alpha' : [float(x) for x in np.linspace(0, 1, num = 11)]
                                }

            # Init
            from xgboost import XGBClassifier
            xg = XGBClassifier(random_state=42)
            xg_tuned = RandomizedSearchCV(xg, hyperparameters, cv=5, random_state=42, scoring='accuracy')
            xg_tuned.fit(X_train,y_train)

            # Predict & Evaluation
            st.write(eval_classification(xg_tuned))
            st.write(eval_cv_ab_accuracy(xg_tuned))
            #st.write(show_best_hyperparameter(xg_tuned))
            
            from sklearn.metrics import ConfusionMatrixDisplay
            # Confusion matrix
            st.write("Result Training")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    xg_tuned,
                    X_train,
                    y_train,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            st.write("Result Test")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    xg_tuned,
                    X_test,
                    y_test,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            st.header("Feature importance")
            fig, axes = plt.subplots(ncols=1, figsize=(20,8))
            feat_importances = pd.Series(xg_tuned.best_estimator_.feature_importances_, index=X.columns)
            ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 8))
            ax.invert_yaxis()
            plt.xlabel('score')
            plt.ylabel('feature')
            plt.title('feature importance score')
            st.pyplot(plt)
            
            st.header("Shap Value")
            import shap
            fig, axes = plt.subplots(ncols=1, figsize=(20,8))
            xg_tuned.fit(X_train,y_train)
            best_model = xg_tuned.best_estimator_
            explainer = shap.Explainer(best_model, X_train)
            shap_values = explainer(X_test)
            feature_importance = np.abs(shap_values.values).mean(axis=0)

            # Create a DataFrame with feature names and importance scores
            feature_importance_df = pd.DataFrame(
                {'Feature': X.columns, 'Importance': feature_importance}
            ).sort_values(by='Importance', ascending=False)

            # Print the feature importance
            #st.write(feature_importance_df)
            shap.summary_plot(shap_values, X_test)
            st.pyplot(plt)
                    
        elif option3 == 'Random Forest':
            from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
            import numpy as np

            # List of hyperparameter
            max_depth = [int(x) for x in np.linspace(1, 10, num = 10)] # Maximum number of levels in tree
            min_samples_split = [2, 5, 10, 100] # Minimum number of samples required to split a node 2, 5, 10, 100
            min_samples_leaf = [1, 2, 4, 10, 20, 50] # Minimum number of samples required at each leaf node 1, 2, 4, 10, 20, 50
            max_features = ['auto', 'sqrt'] # Number of features to consider at every split
            criterion = ['gini','entropy']

            hyperparameters = dict(max_depth=max_depth, 
                                   min_samples_split=min_samples_split, 
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   criterion=criterion
                                  )

            # Init
            from sklearn.ensemble import RandomForestClassifier
            rfc = RandomForestClassifier(random_state=42)
            rfc.fit(X_train, y_train)
            rfc_tuned = RandomizedSearchCV(rfc, hyperparameters, cv=5, random_state=42, scoring='recall')
            rfc_tuned.fit(X_train,y_train)

            # Predict & Evaluation
            st.write(eval_classification(rfc_tuned))
            st.write(eval_cv_ab_roc_auc(rfc))
            st.write(eval_cv_ab_precision(rfc))
            st.wrtite(eval_cv_ab_recall(rfc))

            from sklearn.metrics import ConfusionMatrixDisplay
            # Confusion matrix
            st.write("Result Training")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    rfc_tuned,
                    X_train,
                    y_train,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            st.write("Result Test")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    rfc_tuned,
                    X_test,
                    y_test,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)

            
        elif option3 == 'CATBoost':
            from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
            import numpy as np

            # List of hyperparameter
            hyperparameters = {'depth'         : [4,5,6,7,8,9, 10],
                               'learning_rate' : [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
                               'iterations'    : [10, 20,30,40,50,60,70,80,90, 100],
                               'rsm'           : [0.1,0.2,0.3,0.4,0.5]
                              }

            # Init
            from catboost import CatBoostClassifier
            cg = CatBoostClassifier(silent=True)
            cg_tuned = RandomizedSearchCV(cg, hyperparameters, cv=5, random_state=42, scoring='recall')
            cg_tuned.fit(X_train, y_train)

            # Predict & Evaluation
            st.write(eval_classification(cg_tuned))
            st.write(eval_cv_ab_roc_auc(cg_tuned))
            st.write(eval_cv_ab_recall(cg_tuned))
            
            from sklearn.metrics import ConfusionMatrixDisplay
            # Confusion matrix
            st.write("Result Training")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    rfc_tuned,
                    X_train,
                    y_train,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            st.write("Result Test")
            fig, axes = plt.subplots(ncols=2, figsize=(20,8))
            titles_options = [
                ("Confusion matrix, without normalization", None, 0),
                ("Normalized confusion matrix", "true", 1),
            ]
            for title, normalize, x in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    rfc_tuned,
                    X_test,
                    y_test,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=axes[x]
                )
                disp.ax_.set_title(title)
                #st.write(title)
                #st.write(disp.confusion_matrix)
            st.pyplot(plt)
            
            st.header("**Feature importance**")
            fig, axes = plt.subplots(ncols=1, figsize=(20,8))
            feat_importances = pd.Series(rfc_tuned.best_estimator_.feature_importances_, index=X.columns)
            ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 8))
            ax.invert_yaxis()
            plt.xlabel('score')
            plt.ylabel('feature')
            plt.title('feature importance score')
            st.pyplot(plt)
        else:   
            st.write('Pilih opsi Model terlebih dahulu')
    else:
        st.write('#### Data belum diinput.')
    
        
    