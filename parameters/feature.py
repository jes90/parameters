#!/usr/bin/env python
# coding: utf-8

# In[6]:


from skimage.measure import label, regionprops
import pandas as pd
from skimage import morphology
import numpy as np

class feature:
    def __init__(self,image):
        self.image=image

    def param(self):
        clean_60 = morphology.remove_small_objects(self.image == 255, 8)
        stack_label_60 = label(clean_60,neighbors=8,background=0,connectivity=1)
        regions_60=regionprops(stack_label_60)
        label_number=np.arange(np.max(stack_label_60)+1)
        print(label_number)

        volume=[]
        eccentricity=[]
        eq_diam=[]
        maj_length=[]
        min_length=[]
        bulk=[]
        centroid=[]
        area=[]
        bobox=[]
        theta=[]
        theta_x=[]
        theta_y=[]
        tensor=[]
        for props in regions_60:
            volume.append(props.area)
            eq_diam.append(props.equivalent_diameter)
            bulk.append(props.extent)
            maj_length.append(props.major_axis_length)
            min_length.append(props.minor_axis_length)
            centroid.append(props.centroid)
            bobox.append(props.bbox)
            tensor.append(props.inertia_tensor)
        for i in range(len(tensor)):
            evals,evecs=np.linalg.eig(tensor[i])
            sort_indices = np.argsort(evals)[::-1]
            z_v1,y_v1, x_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
            theta.append(abs(np.arctan((sqrt(y_v1**2+x_v1**2))/(z_v1))))
            theta_x.append(abs(np.arccos(x_v1/(sqrt(z_v1**2+y_v1**2+x_v1**2)))))
            theta_y.append(abs(np.arccos(y_v1/(sqrt(z_v1**2+y_v1**2+x_v1**2)))))
        tables = pd.DataFrame(np.column_stack((volume,
                                               eq_diam,
                                               bulk,
                                               eq_diam,
                                               min_length,
                                               maj_length,
                                               eq_diam,
                                               theta,
                                               theta_x,
                                               theta_y,
                                               centroid,
                                               bobox)))
        df_60=tables.rename(columns={0:"V",
                                     1:"Eq_diam",
                                     2:"B",
                                     3:"E^2",
                                     4:"b",
                                     5:"a",
                                     6:"Q",
                                     7:"Theta",
                                     8:"Theta_x",
                                     9:"Theta_y",
                                     10:"centroid_z",
                                     11:"centroid_y",
                                     12:"centroid_x"})


        df_60['Q']=df_60.b/df_60.a
        df_60['E^2']=1-(df_60.b/df_60.a)**2
        #df_60['T']=(pi*df_60.a**3/3-df_60.V)/df_60.V

        return df_60,label_number,stack_label_60,regions_60

