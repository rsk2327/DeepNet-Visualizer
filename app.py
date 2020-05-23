# import jupyterlab_dash
import dash
import dash_html_components as html
import pandas as pd
import numpy as np
import os
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html

import base64

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from dash.dependencies import Input, Output,State
import dash_core_components as dcc

import torch
import torchvision.transforms as transforms
import torch
from torchvision.transforms import Resize, ToTensor, Normalize
import torch.nn as nn

from PIL import Image

import flask

# from flask.ext.cache import Cache

# cache = Cache(app.server, config={
# 'CACHE_TYPE': 'simple'
# })

# cache.clear()


# Contains CSS info on how to style the different elements in your dashboard
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

containerStyle = {'box-shadow': '2px 2px 2px lightgrey','padding': '15px','background-color': '#f9f9f9'}

bottomMargin = '1.0em'


app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
               assets_folder='/Users/roshansk/Documents/Projects/CovidImaging/assets',
               assets_url_path='/',
               include_assets_files = False)



app.layout = html.Div(children=[
    html.H1(children='DL Net Visualizer'),
    
    html.Div([
        
        html.Div([
            html.Label('Model Folder'),
            dcc.Input(value='', type='text', id='modelFolderInput', style = {'marginBottom': bottomMargin}),
            
            html.Label('Image Folder'),
            dcc.Input(value='', type='text', id = 'imageFolderInput', style = {'marginBottom': bottomMargin}),
                        
            html.Button('Initialize', id='initButton', n_clicks=0, style = {'marginBottom': bottomMargin}),

            html.Label('Model'),
            dcc.Dropdown(id = 'modelInput',value='', style = {'marginBottom': bottomMargin}),
            
            html.Label('Image'),
            dcc.Dropdown(id = 'imageInput',value='', style = {'marginBottom': bottomMargin}),
          
            html.Button('Load', id='loadButton', n_clicks=0, style = {'marginBottom': bottomMargin}),
            
            html.Label('Layer'),
            dcc.Dropdown(id = 'layerInput',value='', style = {'marginBottom': bottomMargin})
            
            
            
            
        ],className='three columns container', style = containerStyle),
        
        html.Div([
            html.Img(id='image', src = 'arr.jpg', style ={'height':'80%'})
            
        ], className = 'nine columns ' ),
        
        
        html.Div(id='testOut')
        
        
        
        
    ])
    
    
    
    

   
])



@app.callback(
    [Output(component_id='modelInput', component_property='options'),
     Output(component_id='imageInput', component_property='options')],
    [Input(component_id='initButton', component_property='n_clicks')],
     [State(component_id='modelFolderInput', component_property='value'),
     State(component_id='imageFolderInput', component_property='value')]
)
def update_model_options(nClicks, modelFolderInput, imageFolderInput):
    
    
    if modelFolderInput == '':
        modelOptions = []
    else:
        fileList = os.listdir(modelFolderInput)

        modelOptions = []

        for i in fileList:
            if ".pt" in i:
                modelOptions.append({'label':i, 'value':i})
    
    if imageFolderInput =='':
        imgOptions = []
    else:
        fileList = os.listdir(imageFolderInput)

        imgOptions = []

        for i in fileList:
            if '.jpg' in i or '.png' in i or 'jpeg' in i:
                imgOptions.append({'label':i, 'value':i})
                
    folder = os.path.join(os.getcwd(),'actData')
    fileList = os.listdir(folder)
    for i in range(len(fileList)):
        os.system(f"rm {os.path.join(folder,fileList[i])}")
        
    
    
    return modelOptions, imgOptions






@app.callback(
     Output(component_id='layerInput', component_property = 'options'),
    [Input(component_id='loadButton', component_property='n_clicks')],
    [State(component_id='modelInput', component_property='value'),
     State(component_id='imageInput', component_property='value'),
     State(component_id='modelFolderInput', component_property='value'),
     State(component_id='imageFolderInput', component_property='value')]
)
def update_activations(nClicks, model, image, modelFolder, imageFolder):
    
    if nClicks == 0:
        return [{'value':1,'label':2},{'value':2,'label':3}]

    folder = os.path.join(os.getcwd(),'actData')
    fileList = os.listdir(folder)
    for i in range(len(fileList)):
        os.system(f"rm {os.path.join(folder,fileList[i])}")

    print("HERE!!!")
    
    net = torch.load(os.path.join(modelFolder, model), map_location=torch.device('cpu'))
    
    img =  Image.open(os.path.join(imageFolder, image)).convert('RGB')

    
    imgSize = 256
    trainTransforms = transforms.Compose([Resize( (imgSize, imgSize) ), ToTensor(),
                                          Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ])
    img1 = trainTransforms(img)
    

    
    x = img1.unsqueeze(0)
    
    activations = {}
    
    ## Extracting activations
    for name, module in net.named_children():
    
        if 'layer' in name:        
            for name_,layer in module.named_children():
                x = layer(x)
                activations[name+"_"+name_] = x.detach().cpu().numpy()

        else:
            x = module(x)
            activations[name] = x.detach().cpu().numpy()

            if 'avgpool' ==name:
                x= x.view(x.size(0),-1)
                
                
    outFile = os.path.join(os.getcwd(),'actData')
    pickleFile = open( os.path.join(outFile,"activationData.pkl" ), "wb")
    pickle.dump(activations, pickleFile)
    pickleFile.close()
                
    
    
    layerOptions = []
    
    for key in list(activations.keys()):
        layerOptions.append({'value':key,'label':key})

    return  layerOptions


@app.callback(
Output('image','src'),
[Input('layerInput','value')])
def displayFigure(layerInput):

    if layerInput =='':
        return None
    
    print(layerInput)
    filename = os.path.join(os.getcwd(),'actData', f"act_{layerInput}.jpg")

    if os.path.exists(filename):
        return f"act_{layerInput}.jpg"
    else:
        pickleFile = os.path.join(os.getcwd(),'actData','activationData.pkl')
        activationData = pickle.load( open( pickleFile, "rb" ) )

        act = activationData[layerInput].squeeze(0)

        plt.figure(figsize = (15,15))

        for i in range(6):
            for j in range(6):
                plt.subplot(6,6,i*6 + j + 1)
                plt.imshow(act[i*6 + j, : , :], cmap='gray')
                plt.axis('off')
                
                
        

        filename = os.path.join(os.getcwd(),'actData', f"act_{layerInput}.jpg")
        plt.savefig(filename, bbox_inches='tight', pad_inches = 0)
        plt.close()
        

        encoded_image = base64.b64encode(open(filename, 'rb').read())
        
        src='data:image/jpg;base64,{}'.format(encoded_image.decode())

        return src



@app.callback(
    Output('testOut', 'children'),
    [Input('initButton', 'n_clicks')])
def update_outout(value):
    print((flask.request.cookies))
    return value
    



if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload = False, port=9000)

# @app.callback(
#     dash.dependencies.Output('image', 'src'),
#     [dash.dependencies.Input('image-dropdown', 'value')])
# def update_image_src(value):
#     return static_image_route + value