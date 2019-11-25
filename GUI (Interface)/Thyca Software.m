function ThycaGuiHack2019
% Main guiFig construction
guiFig = figure('DockControls','off','Resize','off','Units','pixels',...
    'Position',[50 200-100 600 500],'MenuBar','none','Name','Thyca Package',...
    'ToolBar','none','Visible','on','NumberTitle','off','UserData',0,...
    'Interruptible','off');%,'CloseRequestFcn',@figureClose);
defaultBackground = get(0,'defaultUicontrolBackgroundColor');

set(guiFig,'Color',defaultBackground);
handles.output = guiFig;
guidata(guiFig,handles);

%Create tab group
handles.tgroup = uitabgroup('Parent', guiFig,'TabLocation', 'top');
handles.tab1 = uitab('Parent', handles.tgroup, 'Title', 'Analyze Ultrasonic Data');
handles.tab2 = uitab('Parent', handles.tgroup, 'Title', 'Add to Thyca Database');

%tab1
% loading image 
handles.Load1 = uicontrol('Parent',handles.tab1,'Style', 'pushbutton',...
    'String','Load image','Enable','on','Position', [50 330 120 40],'Callback',@load1Fn);
handles.ImageView1 = uicontrol('Parent',handles.tab1,'Style', 'Text',...
    'String','Image view','Enable','on','Position', [18 265 120 40]);
handles.ImageView2 = uicontrol('Parent',handles.tab1,'Style', 'popupmenu',...
    'String',{ ' Trans',' Sag'} ...
    ,'Position', [120 268 50 40]);

% Adding Image axes
handlesIm.axes1 = axes('Parent',handles.tab1,'Units','pixels','Visible','off',...
    'Position',[230 125 300 300]);
% Adding Image axes
% handlesIm.axes2 = axes('Parent',handles.tab1,'Units','pixels','Visible','on',...
%     'Position',[630 125 300 300]);

% % select the area 
% handles.SelectROI1 = uicontrol('Parent',handles.tab1,'Style', 'pushbutton', ...
%     'String','Select the area','Enable','on','Position', [50 230 120 40],...
%     'Callback',@selectFn);
% anaylze input ultrasonic data  
handles.Ana1 = uicontrol('Parent',handles.tab1,'Style', 'pushbutton',...
    'String','Analyze','Enable','on','Position', [50 230 120 40],'Callback',...
    @AnaFn);

handles.BeningText = uicontrol('Parent',handles.tab1,'Style', 'text',...
    'String','Benign (%)','Position', [25 175 80 40]);

handles.NonBeningText = uicontrol('Parent',handles.tab1,'Style', 'text',...
    'String','Non-Benign (%)','Position', [25 165 80 15]);

handles.BeningEdit = uicontrol('Parent',handles.tab1,'Style', 'text',...
    'Position', [120 195 40 25]);

handles.NonBeningEdit = uicontrol('Parent',handles.tab1,'Style', 'text',...
   'Position', [120 160 40 25]);

%save results as a text file
handles.SaveResult = uicontrol('Parent',handles.tab1,'Style', 'pushbutton',...
    'String','Save results','Enable','on','Position', [450 30 120 40],'Callback',@saveFn);

%tab2
% loading image 
handles.Load2 = uicontrol('Parent',handles.tab2,'Style', 'pushbutton',...
    'String','Load data','Enable','on','Position', [50 330 120 40],...
    'Callback',@load2Fn);
% % select the area 
% handles.Crop = uicontrol('Parent',handles.tab2,'Style', 'pushbutton', ...
%     'String','Crop the area','Enable','on','Position', [50 230 120 40]);
% Adding Image axes
handlesIm.axes2 = axes('Parent',handles.tab2,'Units','pixels','Visible','off',...
    'Position',[230 125 300 300]);
% add more information
%Thyrod Category
handles.TyrCatText = uicontrol('Parent',handles.tab2,'Style', 'text',...
    'String','Thyroid Category','Enable','on','Position', [15 265 120 40]);
handles.TyrCatList = uicontrol('Parent',handles.tab2,'Style', 'popupmenu',...
    'String',{ ' N',' O:MNG',' O:Suspecious',' T2',' T3',' T4',' T5'} ...
    ,'Position', [120 268 50 40]);
%FNA Dignosis
handles.FNAText = uicontrol('Parent',handles.tab2,'Style', 'text',...
    'String','FNA Diagnosis','Position', [15 230 120 40]);
handles.FNAEdit = uicontrol('Parent',handles.tab2,'Style', 'edit',...
    'String','1','Position', [120 249 50 25]);

% add the image 
handles.AddDataButton = uicontrol('Parent',handles.tab2,'Style', 'pushbutton',...
    'String','Add database','Position', [50 195 120 40],'Callback',@addFn);


%%

function load1Fn(~,~)
[Filename filedir]=uigetfile('*.png')
I=imread(fullfile(filedir,Filename));
 
 imagesc(handlesIm.axes1,I);
 colormap(gray(256))
[J]=imcrop(I);

imagesc(handlesIm.axes1,J);
 colormap(gray(256))
end 

function load2Fn(~,~)
[Filename filedir]=uigetfile('*.png')
I=imread(fullfile(filedir,Filename));
 
 imagesc(handlesIm.axes2,I);
 colormap(gray(256))
 
[J]=imcrop(I);
imagesc(handlesIm.axes2,J);
 colormap(gray(256))
end 

function selectFn(~,~)
end 

function AnaFn(~,~)
ProbBenign='75';
ProbNonBenign='25';
set(handles.Ana1,'BackgroundColor','r')
pause(2)
set(handles.NonBeningEdit, 'BackgroundColor','r')
set(handles.NonBeningEdit,'String',ProbBenign,'FontWeight','Bold','FontSize',14)
set(handles.BeningEdit, 'BackgroundColor','g')
set(handles.BeningEdit,'String',ProbNonBenign,'FontWeight','Bold','FontSize',14)
set(handles.Ana1,'BackgroundColor',[.94 .94 .94])

end 

    function saveFn(~,~)
        [fileDir]=uigetdir(pwd)
    end 

    function addFn(~,~) 
        set(handles.AddDataButton,'BackgroundColor','r')
        pause(1)
        set(handles.AddDataButton,'BackgroundColor',[.94 .94 .94])

    end 
end 