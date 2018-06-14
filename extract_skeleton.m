%% author: Shaoju Wu Date: 5/29/2018 extract the skeleton voxel from the data
meanSkeleton=readFileNifti(['mean_FA_skeleton_mask.nii.gz']);
mask=meanSkeleton.data;
stackInfo.PixelSpacing = [1 1];
stackInfo.SpacingBetweenSlices = 1;
%find the non-zero pixel location
index=find(mask==1);
%% Visualize skeleton mask
b=visibleCube(mask,stackInfo, .5);
b.FaceAlpha=0.2;
b.FaceColor='g';
%% Loading FA image to extract skeleton voxel data
% fileFolder='/media/shaoju/Data/3D_CNN/TBSS_v486/mytbss/Registration_to_MNI/fa_skeleton/';
% dirOutput=dir(fullfile(fileFolder,'*.nii.gz'));
% % initialize 
% FA_image=zeros(249,length(index));
% fileNamesFA={dirOutput.name}';
% for i=1:length(fileNamesFA)
%     im=readFileNifti([fileFolder,fileNamesFA{i}]);
%     FA_image(i,:)=im.data(index);
%     
% end
% %% Loading MD image to extract skeleton voxel data
% fileFolder='/media/shaoju/Data/3D_CNN/TBSS_v486/mytbss/Registration_to_MNI/md_skeleton/';
% dirOutput=dir(fullfile(fileFolder,'*.nii.gz'));
% % initialize 
% MD_image=zeros(249,length(index));
% fileNamesMD={dirOutput.name}';
% for i=1:length(fileNamesMD)
%     im=readFileNifti([fileFolder,fileNamesMD{i}]);
%     MD_image(i,:)=im.data(index);
% end
% %% Loading AD image to extract skeleton voxel data
% fileFolder='/media/shaoju/Data/3D_CNN/TBSS_v486/mytbss/Registration_to_MNI/ad_skeleton/';
% dirOutput=dir(fullfile(fileFolder,'*.nii.gz'));
% % initialize 
% AD_image=zeros(249,length(index));
% fileNamesAD={dirOutput.name}';
% for i=1:length(fileNamesAD)
%     im=readFileNifti([fileFolder,fileNamesAD{i}]);
%     AD_image(i,:)=im.data(index);
% end
%% Loading RD image to extract skeleton voxel data
fileFolder='/media/shaoju/Data/3D_CNN/TBSS_v486/mytbss/Registration_to_MNI/rd_skeleton/';
dirOutput=dir(fullfile(fileFolder,'*.nii.gz'));
% initialize 
RD_image=zeros(249,length(index));
fileNamesRD={dirOutput.name}';
for i=1:length(fileNamesRD)
    im=readFileNifti([fileFolder,fileNamesRD{i}]);
    RD_image(i,:)=im.data(index);
end

%% loading Training data from control group and moderate concussion group
currentdirectory = pwd;
% path of control group
fileFolder=[currentdirectory,'/genctrl/fa/'];
fileFolder_concussion=[currentdirectory,'/modTBI/fa/'];
%MD modality
fileFolder_MD=[currentdirectory,'/genctrl/md/'];
fileFolder_concussion_MD=[currentdirectory,'/modTBI/md/'];

% path of concussion group
dirOutput=dir(fullfile(fileFolder,'*.nii.gz'));
dirOutput_concussion=dir(fullfile(fileFolder_concussion,'*.nii.gz'));
% MD modality
dirOutput_MD=dir(fullfile(fileFolder_MD,'*.nii.gz'));
dirOutput_concussion_MD=dir(fullfile(fileFolder_concussion_MD,'*.nii.gz'));
% initialize 
FA_data=zeros(24,length(index));
MD_data=zeros(24,length(index));
label=zeros(24,1);
fileNames={dirOutput.name}';
fileNames_concussion={dirOutput_concussion.name}';

fileNames_MD={dirOutput_MD.name}';
fileNames_concussion_MD={dirOutput_concussion_MD.name}';

for i=1:length(fileNames)
    im=readFileNifti([fileFolder,fileNames{i}]);
    im_con=readFileNifti([fileFolder_concussion,fileNames_concussion{i}]);
    
    im_MD=readFileNifti([fileFolder_MD,fileNames_MD{i}]);
    im_con_MD=readFileNifti([fileFolder_concussion_MD,fileNames_concussion_MD{i}]);
    
    FA_data(i,:)=im.data(index);
    MD_data(i,:)=im_MD.data(index);
    label(i)=0;
    FA_data(i+12,:)=im_con.data(index);
    MD_data(i+12,:)=im_con_MD.data(index);
    label(i+12)=1;
    i
end

%% Visualize skeleton mask
% h=visibleCube(im.data,stackInfo, .1);
% h.FaceAlpha=0.2; 