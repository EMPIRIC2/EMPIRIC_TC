# -*- coding: utf-8 -*-
"""
@author: Nadia Bloemendaal, nadia.bloemendaal@vu.nl

This script calculates the changes in the STORM variables between the global climate models' present- and future- climate datasets. This is de so-called delta that is then
added to the IBTrACS statistics and that will then serve as input for the STORM model, to generate the future-climate STORM datasets with.

This script is part of the STORM Climate change research. Please read the corresponding paper before commercing.
Bloemendaal et al (2022) A globally consistent local-scale assessment of future tropical cyclone risk. Paper published in Science Advances.

Copyright (C) 2020 Nadia Bloemendaal. All versions realeased under the GNU General Public License v3.0.
"""

import numpy as np
import os
import sys
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
import pandas as pd 
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

#Set the location to where you want to store the new files. Current setting = current working directory.
dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

land_shp_fname = shpreader.natural_earth(resolution='50m',category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def is_land(x, y):
    return land.contains(sgeom.Point(x, y))

def create_mask(basin):
    stepsize=10
    lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(basin)
    x=int(abs(lon1-lon0)*stepsize)
    y=int(abs(lat1-lat0)*stepsize)
    if lon0<180: #south pacific
        lon_grid,lat_grid=np.mgrid[lon0:lon1:complex(0,x),lat0:lat1:complex(0,y)]        
    else:  
        lon_grid,lat_grid=np.mgrid[lon0-360:lon1-360:complex(0,x),lat0:lat1:complex(0,y)]
    
    mask=np.ones((len(lon_grid[0]),len(lon_grid)))
    for i in range(len(lon_grid)):
        for j in range(len(lon_grid[i])):
            mask[j][i]=is_land(lon_grid[i][j],lat_grid[i][j])

    mask=np.flipud(mask)
    
    return mask

def shift_normal_distribution(mu_present,mu_future,mu_baseline,var_present,var_future,var_baseline):
    """
    Parameters
    ----------
    mu_present : mean for present-climate dataset
    mu_future : mean for future-climate dataset
    mu_baseline : mean for baseline dataset, on which the delta will be added
    var_present : variance for present-climate dataset
    var_future : variance for future-climate dataset
    var_baseline : variance for baseline dataset, on which the delta will be added

    Returns
    -------
    mu_shifted: the new mu 
    var_shifted: the new variance

    """
    mu_shifted=mu_baseline+(mu_future-mu_present)
    var_shifted=var_baseline*(var_future/var_present)
    
    return mu_shifted,var_shifted

def BOUNDARIES_BASINS(idx):
    if idx=='EP': #Eastern Pacific
        lat0,lat1,lon0,lon1=5,60,180,285
    if idx=='NA': #North Atlantic
        lat0,lat1,lon0,lon1=5,60,255,359
    if idx=='NI': #North Indian
        lat0,lat1,lon0,lon1=5,60,30,100
    if idx=='SI': #South Indian
        lat0,lat1,lon0,lon1=-60,-5,10,135
    if idx=='SP': #South Pacific
        lat0,lat1,lon0,lon1=-60,-5,135,240
    if idx=='WP': #Western Pacific
        lat0,lat1,lon0,lon1=5,60,100,180
    
    return lat0,lat1,lon0,lon1

def create_5deg_grid(locations,month,basin): #this function creates the 5-degree grid for the genesis locations (see also the STORM preprocessing scripts)
    step=5.

    lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(basin)
    if basin=='NA':    
        lonspace=np.linspace(lon0,360.,int(abs(lon0-360.)/step)+1)
    else:
        lonspace=np.linspace(lon0,lon1,int(abs(lon0-lon1)/step)+1)
    
    latspace=np.linspace(lat0,lat1,int(abs(lat0-lat1)/step)+1)   
    
    lat_list=[locations[month][i][0] for i in range(len(locations[month])) if (lat0<=locations[month][i][0]<=lat1 and lon0<=locations[month][i][1]<=lon1)]
    lon_list=[locations[month][i][1] for i in range(len(locations[month])) if (lat0<=locations[month][i][0]<=lat1 and lon0<=locations[month][i][1]<=lon1)]
    
    df=pd.DataFrame({'Latitude':lat_list,'Longitude':lon_list})
    
    to_bin=lambda x:np.floor(x/step)*step
    df["latbin"]=df.Latitude.map(to_bin)
    df["lonbin"]=df.Longitude.map(to_bin)
    groups=df.groupby(["latbin","lonbin"])
    count_df=pd.DataFrame({'count':groups.size()}).reset_index()
    counts=count_df["count"]       
    latbin=groups.count().index.get_level_values('latbin')
    lonbin=groups.count().index.get_level_values('lonbin')
    count_matrix=np.zeros((len(latspace),int(abs(lon0-lon1)/step)+1))
    
    for lat,lon,count in zip(latbin,lonbin,counts):
          i=latspace.tolist().index(lat)
          j=lonspace.tolist().index(lon)
          count_matrix[i,j]=count
    
    return count_matrix
  
def create_1deg_grid(delta_count_matrix,basin,month): #this function creates the 1-degree genesis grid (see also STORM preprocessing scripts)
    step=5.
    
    lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(basin)

    latspace=np.linspace(lat0,lat1,int(abs(lat0-lat1)/step)+1)
    lonspace=np.linspace(lon0,lon1,int(abs(lon0-lon1)/step)+1)
            
    xg=int(abs(lon1-lon0))
    yg=int(abs(lat1-lat0))
    xgrid,ygrid=np.mgrid[lon0:lon1:complex(0,xg),lat0:lat1:complex(0,yg)]
    points=[]
    for i in range(len(lonspace)):
        for j in range(len(latspace)):
            points.append((lonspace[i],latspace[j]))
     
    values=np.reshape(delta_count_matrix.T,int(len(lonspace))*int(len(latspace)))
    grid=griddata(points,values,(xgrid,ygrid),method='cubic')
    grid=np.transpose(grid)
    grid=np.flipud(grid)
    grid[grid<0]=0
            

    #overlay data with a land-sea mask
    mdata=create_mask(basin)
    coarseness=10
    mdata_coarse=mdata.reshape((mdata.shape[0]//coarseness,coarseness,mdata.shape[1]//coarseness,coarseness))
    mdata_coarse=np.mean(mdata_coarse,axis=(1,3))
                         
    (x,y)=mdata_coarse.shape
    
    for i in range(0,x):
        for j in range(0,y):
            if mdata_coarse[i,j]>0.50:
                grid[i,j]='nan'        

    return grid

#The functions below will calculate the shift in the STORM input variables based on the information from the global climate models. The output files are thus the shifted
#STORM variables and represent the future-climate conditions (and serve as input for the STORM model).


# =============================================================================
# Step 1: Calculate the relative change in genesis frequency
# =============================================================================
def Change_genesis_frequency(model):
    dataset_name='POISSON_GENESIS_PARAMETERS'
    print(dataset_name)
    delta_poisson=[]
    present=np.load(os.path.join(__location__,'{}_PRESENT_{}_nothres.npy'.format(dataset_name,model)),allow_pickle=True,encoding='latin1').item()
    future=np.load(os.path.join(__location__,'{}_FUTURE_{}_nothres.npy'.format(dataset_name,model)),allow_pickle=True,encoding='latin1').item()
    
    ibtracs=np.loadtxt(os.path.join(__location__,dataset_name+'.txt'))
    
    delta_poisson=[]
    
    for basin,basinidx in zip(['EP','NA','NI','SI','SP','WP'],range(0,6)):
        delta_poisson.append((future[basin][0]/present[basin][0])*ibtracs[basinidx])
        print(model,basin,ibtracs[basinidx],round(delta_poisson[basinidx],1))

    np.savetxt(os.path.join(__location__,'{}_IBTRACSDELTA_{}.txt'.format(dataset_name,model)),delta_poisson)
    
    del dataset_name, present, future

# =============================================================================
# Step 2: Calculate the relative change in genesis months
# This is given as the difference between the number of genesis per month for 
# present and future, divided by the total number of formations in the present climate
# =============================================================================
def Change_genesis_month(model):
    dataset_name='GENESIS_MONTHS'
    print(dataset_name)
    present_all=np.load(os.path.join(__location__,'{}_PRESENT_{}_nothres.npy'.format(dataset_name,model)),allow_pickle=True,encoding='latin1').item()
    future_all=np.load(os.path.join(__location__,'{}_FUTURE_{}_nothres.npy'.format(dataset_name,model)),allow_pickle=True,encoding='latin1').item()

    #also open the ibtracs dataset here
    ibtracs_all = np.load(os.path.join(__location__,dataset_name+'.npy'),allow_pickle=True,encoding='latin1').item()
    
    genesis_months={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
    monthsall={'EP':[6,7,8,9,10,11],'NA':[6,7,8,9,10,11],'NI':[4,5,6,9,10,11],'SI':[1,2,3,4,11,12],'SP':[1,2,3,4,11,12],'WP':[5,6,7,8,9,10,11]}
   
    for basin,basinidx in zip(['EP','NA','NI','SI','SP','WP'],range(0,6)):   
        ibtracs=ibtracs_all[basinidx]
        present=present_all[basin]
        future=future_all[basin]
        
        ibtracs=np.array(ibtracs)
        present=np.array(present)
        future=np.array(future)
                
        for month in monthsall[basin]:
            
            month_present=np.count_nonzero(present==month)
            month_future=np.count_nonzero(future==month)
            #also count the month_ibtracs here
            month_ibtracs=np.count_nonzero(ibtracs==month)
                
            delta=(month_future-month_present)/len(present)
            
            #assume delta is constant, so that
            delta_month=month_ibtracs + delta*len(ibtracs)
            
            for i in range(int(delta_month)):
                genesis_months[basin].append(int(month))

    np.save(os.path.join(__location__,'{}_IBTRACSDELTA_{}.npy'.format(dataset_name,model)),genesis_months)
        
    #del dataset_name,present,future

#%%
# =============================================================================
# Step 4: Genesis pressure
# For this, we need to shift the normal distribution of genesis pressure in the present climate
# to the future climate. 
# =============================================================================
def Change_genesis_pressure(model):
    
    dataset_name='DP0_PRES_GENESIS'
    print(dataset_name)
    present_all=np.load(os.path.join(__location__,'{}_PRESENT_{}_nothres.npy'.format(dataset_name,model)),
                    allow_pickle=True,encoding='latin1').item()
    future_all=np.load(os.path.join(__location__,'{}_FUTURE_{}_nothres.npy'.format(dataset_name,model)),
                   allow_pickle=True,encoding='latin1').item()
    ibtracs_all=np.load(os.path.join(__location__,'{}.npy'.format(dataset_name)),
                    allow_pickle=True,encoding='latin1').item()
    
        
    monthsall={'EP':[6,7,8,9,10,11],'NA':[6,7,8,9,10,11],'NI':[4,5,6,9,10,11],'SI':[1,2,3,4,11,12],'SP':[1,2,3,4,11,12],'WP':[5,6,7,8,9,10,11]}
    
    adjusted={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
   
    for basin,basinidx in zip(['EP','NA','NI','SI','SP','WP'],range(0,6)):

        ibtracs=ibtracs_all[basinidx]
        present=present_all[basin]
        future=future_all[basin]

        adjusted[basin]={i:[] for i in monthsall}
        
        for month in monthsall[basin]:
            mupres_present,varpres_present=present[month][0],present[month][1]**2.
            mupres_future,varpres_future=future[month][0],future[month][1]**2.
            
            mupres_ibtracs,varpres_ibtracs=ibtracs[month][0],ibtracs[month][1]**2.
            
            mupres_adj,varpres_adj=shift_normal_distribution(mupres_present,mupres_future,mupres_ibtracs,
                                                             varpres_present,varpres_future,varpres_ibtracs)
            
           
            mu_present,var_present=present[month][2],present[month][3]**2.
            mu_future,var_future=future[month][2],future[month][3]**2.
            mu_ibtracs,var_ibtracs=ibtracs[month][0],ibtracs[month][1]**2.
         
            mu_adj,var_adj=shift_normal_distribution(mu_present,mu_future,mu_ibtracs,
                                                     var_present,var_future,var_ibtracs)
            
            dpmin_present,dpmax_present=present[month][4],present[month][5]
            dpmin_future,dpmax_future=future[month][4],future[month][5]
            dpmin_ibtracs,dpmax_ibtracs=ibtracs[month][4],ibtracs[month][5]   
            dpmin_adj=dpmin_ibtracs*(dpmin_future-dpmin_present)/dpmin_present+dpmin_ibtracs
            dpmax_adj=dpmax_ibtracs*(dpmax_future-dpmax_present)/dpmax_present+dpmax_ibtracs
            
            adjusted[basin][month]=[mupres_adj,np.sqrt(varpres_adj),mu_adj,np.sqrt(var_adj),dpmin_adj,dpmax_adj]
        
    
    np.save(os.path.join(__location__,'{}_IBTRACSDELTA_{}.npy'.format(dataset_name,model)),adjusted)
    
    del dataset_name,present,future,ibtracs

#%%
# =============================================================================
# Step 5: shift the lon/lat coefficients 
# =============================================================================
def Change_longitude_latitude(model):
    dataset_name='JM_LONLATBINS'
    print(dataset_name)
    present_all=np.load(os.path.join(__location__,'{}_PRESENT_{}_nothres.npy'.format(dataset_name,model)),allow_pickle=True,encoding='latin1').item()
    future_all=np.load(os.path.join(__location__,'{}_FUTURE_{}_nothres.npy'.format(dataset_name,model)),allow_pickle=True,encoding='latin1').item()
    
    
    adjusted={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
   
    for basin,basinidx in zip(['EP','NA','NI','SI','SP','WP'],range(0,6)):
        ibtracs=np.loadtxt(os.path.join(__location__,dataset_name+'_'+str(basinidx)+'.txt'))
        present=present_all[basinidx]
        future=future_all[basinidx]
        
        for i in range(len(present)):
            lijst=[]
            for j in range(0,5): #the first 5 components are coefficients - do not shift those!
                lijst.append(ibtracs[i][j])
            
            if basin=='NA':
                for j in range(5,13):
                    if ibtracs[i][j]>19.: 
                        lijst.append(0.35)
                    else:
                        lijst.append(ibtracs[i][j])
                print(len(lijst))
                
            else:
                for j in [5,7,9,11]: #the remainder are mu and std for various variables                
                    mu_adj,var_adj=shift_normal_distribution(present[i][j],future[i][j],ibtracs[i][j],
                                                             present[i][j+1]**2.,future[i][j+1]**2.,ibtracs[i][j+1]**2.)
            
                    lijst.append(mu_adj)
                    lijst.append(np.sqrt(var_adj))
                print(len(lijst))
                
            adjusted[basin].append(lijst)
        
                
        np.save(os.path.join(__location__,'{}_IBTRACSDELTA_{}.npy'.format(dataset_name,model)),adjusted)
    
    del present,future,dataset_name

#%%
# =============================================================================
# Step 6: shift the pressure coefficients  
#this part will also shift the mpi! <- this is done as a relative change  
# =============================================================================

def Change_pressure(model):
    #Make sure that the new MPI is not below the MPI threshold:
    mpibounds={'EP':[860,880,900,900,880,860],'NA':[920,900,900,900,880,880],'NI':[840,860,880,900,880,860],'SI':[840,880,860,860,840,860],'SP':[840,840,860,860,840,840],'WP':[860,860,860,870,870,860,860]}
    
    
    dataset_name='COEFFICIENTS_JM_PRESSURE'
    print(dataset_name)
    present_all=np.load(os.path.join(__location__,'{}_{}_PRESENT.npy'.format(dataset_name,model)),allow_pickle=True,encoding='latin1').item()
    future_all=np.load(os.path.join(__location__,'{}_{}_FUTURE.npy'.format(dataset_name,model)),allow_pickle=True,encoding='latin1').item()
    ibtracs_all=np.load(os.path.join(__location__,dataset_name+'.npy'),allow_pickle=True,encoding='latin1').item()
        
    monthsall={'EP':[6,7,8,9,10,11],'NA':[6,7,8,9,10,11],'NI':[4,5,6,9,10,11],'SI':[1,2,3,4,11,12],'SP':[1,2,3,4,11,12],'WP':[5,6,7,8,9,10,11]}
    
    adjusted={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
   
    for basin,basinidx in zip(['EP','NA','NI','SI','SP','WP'],range(0,6)):
        
        adjusted[basin]={i:[] for i in monthsall[basin]}
        
        ibtracs=ibtracs_all[basinidx]
        present=present_all[basin]
        future=future_all[basin]
    
        for month,index in zip(monthsall[basin],range(len(monthsall[basin]))):
            for i in range(len(present[month])):
                lijst=[]
                #first up are the 4 coefficients from the pressure regression formula
                for j in range(0,4):
                    lijst.append(ibtracs[month][i][j])
                
                #the next two items in the list are the mu and std for the random-error term
                mu_adj,var_adj=shift_normal_distribution(present[month][i][4], 
                                                          future[month][i][4], 
                                                          ibtracs[month][i][4], 
                                                          present[month][i][5]**2., 
                                                          future[month][i][5]**2.,
                                                          ibtracs[month][i][5]**2.)
                lijst.append(mu_adj)
                lijst.append(np.sqrt(var_adj))
        
                #the last item is the MPI, which we model as a relative change
                MPI_relchange=(future[month][i][6]-present[month][i][6])/present[month][i][6]
        
                MPI_adj=ibtracs[month][i][6]*MPI_relchange+ibtracs[month][i][6]
                if MPI_adj<mpibounds[basin][index]:
                    MPI_adj=mpibounds[basin][index]
                
                
                lijst.append(MPI_adj)
            
                adjusted[basin][month].append(lijst)
    
    np.save(os.path.join(__location__,'{}_IBTRACSDELTA_{}.npy'.format(dataset_name,model)),adjusted)
    
    del dataset_name,present,future

#%%
# =============================================================================
# Step 7: Adjust the enviromental pressure-fields. Delta = relative change
# =============================================================================
def Change_monthly_MSLP(model):
    dataset_name='Monthly_mean_MSLP'
    print(dataset_name)
    from scipy.ndimage.interpolation import zoom
      
    for month in range(1,13):
        
        present=np.loadtxt(os.path.join(__location__,'{}_{}_{}_PRESENT_flipped.txt'.format(dataset_name,model,month)))
        future=np.loadtxt(os.path.join(__location__,'{}_{}_{}_FUTURE_flipped.txt'.format(dataset_name,model,month)))
        ibtracs=np.loadtxt(os.path.join(__location__,dataset_name+'_'+str(month)+'.txt'))
        
        delta=(future-present)/present        
        
        (latdim,londim)=delta.shape
        (latib,lonib)=ibtracs.shape
        
        delta_regridded=zoom(delta,(latib/latdim,lonib/londim))        
        
        MSLP_adj=ibtracs*delta_regridded+ibtracs
    
        np.savetxt(os.path.join(__location__,'{}_{}_{}_IBTRACSDELTA.txt'.format(dataset_name,model,month)),MSLP_adj)
        
    
    del dataset_name,present,future,MSLP_adj

#%% 
# =============================================================================
# Step 8: create the genesis matrix (from genesis_matrix_relative_changes.py)
# =============================================================================
def Change_genesis_locations(model):
    matrix_dict={i:[] for i in ['PRESENT','FUTURE','IBTRACS']}
    print('Genesis_locations')
    monthsall={'EP':[6,7,8,9,10,11],'NA':[6,7,8,9,10,11],'NI':[4,5,6,9,10,11],'SI':[1,2,3,4,11,12],'SP':[1,2,3,4,11,12],'WP':[5,6,7,8,9,10,11]}
    
    genesis_grids={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
    
    for period in ['PRESENT','FUTURE']:
        matrix_dict[period]={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        
        locations=np.load(os.path.join(__location__,'GEN_LOC_{}_{}_nothres.npy'.format(period,model)),allow_pickle=True,encoding='latin1').item()
    
        for basin in ['EP','NA','NI','SI','SP','WP']:
            matrix_dict[period][basin]={i:[] for i in monthsall[basin]}
            
            for month in monthsall[basin]:
                matrix_dict[period][basin][month]=create_5deg_grid(locations[basin],month,basin)
    
    locations=np.load(os.path.join(__location__,'GEN_LOC.npy'),allow_pickle=True,encoding='latin1').item()
    matrix_dict['IBTRACS']={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
    
    for basin,basinidx in zip(['EP','NA','NI','SI','SP','WP'], [0,1,2,3,4,5]):
        print(basin)
        matrix_dict['IBTRACS'][basin]={i:[] for i in monthsall[basin]}
        genesis_grids[basin]={i:[] for i in monthsall[basin]}
        
        
        #calculate the relative change per month
        absolute_delta={i:[] for i in monthsall[basin]}
        relative_delta={i:[] for i in monthsall[basin]}
        
        for month in monthsall[basin]:
            #matrix_dict['IBTRACS'][basin][month]=create_5deg_grid(locations[basinidx],month,basin)
            ibtracs=create_5deg_grid(locations[basinidx],month,basin)
            
            present=np.array(matrix_dict['PRESENT'][basin][month])
            future=np.array(matrix_dict['FUTURE'][basin][month])
            #ibtracs=np.array(matrix_dict['IBTRACS'][basin][month])
            
            (latlen,lonlen)=present.shape
            
            relative_change=np.zeros((latlen,lonlen))
            absolute_change=np.zeros((latlen,lonlen))
            
            test_dummy=np.zeros((latlen,lonlen))
            
            # lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(basin)
            # step=5
            # if basin=='NA':    
            #     lonspace=np.linspace(lon0,360.,int(abs(lon0-360.)/step)+1)
            # else:
            #     lonspace=np.linspace(lon0,lon1,int(abs(lon0-lon1)/step)+1)
            
            # latspace=np.linspace(lat0,lat1,int(abs(lat0-lat1)/step)+1)
            # latspace=np.flip(latspace)
            
            for i in range(latlen):
                for j in range(lonlen):
                    if present[i][j]>0.: #relative change
                        relative_change[i][j]=(future[i][j]-present[i][j])/present[i][j]
                        absolute_change[i][j]=np.nan
                    else:
                        relative_change[i][j]=np.nan
                        absolute_change[i][j]=future[i][j]-present[i][j]
   
            absolute_delta[month]=absolute_change
            relative_delta[month]=relative_change
          
            #test to see if this is working
            for i in range(latlen):
                for j in range(lonlen):
                    if np.isnan(absolute_delta[month][i][j])==False:
                        #then absolute difference
                        test_dummy[i][j]=ibtracs[i][j]+absolute_delta[month][i][j]
                    else:
                        #then relative difference
                        test_dummy[i][j]=ibtracs[i][j]*relative_delta[month][i][j]+ibtracs[i][j]

            blurred_counts = gaussian_filter(test_dummy, 1)

            grid_genesis=create_1deg_grid(blurred_counts,basin,month)
            genesis_grids[basin][month]=grid_genesis
            
    np.save(os.path.join(__location__,'GENESIS_LOCATIONS_IBTRACSDELTA_{}.npy'.format(model)),genesis_grids)
    

for model in ['CMCC-CM2-VHR4','EC-Earth3P-HR','CNRM-CM6-1-HR','HadGEM3-GC31-HM']:
    print(model)
    #Change_genesis_frequency(model)
    #Change_genesis_month(model)
    #Change_genesis_pressure(model)
    #Change_longitude_latitude(model)
    #Change_monthly_MSLP(model)
    Change_genesis_locations(model)
    #Change_pressure(model)
