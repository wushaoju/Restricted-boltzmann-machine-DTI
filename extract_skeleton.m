%% Extract TBSS skeleton from mildTBI and genctrl cases and label the data
% Author: Shaoju Wu Date: 2018 6 15
meanSkeleton=readFileNifti(['mean_FA_skeleton_mask.nii.gz']);
mask=meanSkeleton.data;
stackInfo.PixelSpacing = [1 1];
stackInfo.SpacingBetweenSlices = 1;
%find the non-zero pixel location
index=find(mask==1);
%% Visualize skeleton mask
% b=visibleCube(mask,stackInfo, .5);
% b.FaceAlpha=0.2;
% b.FaceColor='g';
%% loading Training data from control group and mild concussion group
currentdirectory = pwd;
% path of control group
fileFolder_FA=[currentdirectory,'/genctrl/fa/'];
fileFolder_concussion_FA=[currentdirectory,'/mildTBI/fa/'];

%MD modality
fileFolder_MD=[currentdirectory,'/genctrl/md/'];
fileFolder_concussion_MD=[currentdirectory,'/mildTBI/md/'];

%AD modality
fileFolder_AD=[currentdirectory,'/genctrl/ad/'];
fileFolder_concussion_AD=[currentdirectory,'/mildTBI/ad/'];

%RD modality
fileFolder_RD=[currentdirectory,'/genctrl/rd/'];
fileFolder_concussion_RD=[currentdirectory,'/mildTBI/rd/'];

% path of concussion group
dirOutput_FA=dir(fullfile(fileFolder_FA,'*.nii.gz'));
dirOutput_concussion_FA=dir(fullfile(fileFolder_concussion_FA,'*.nii.gz'));

% MD modality
dirOutput_MD=dir(fullfile(fileFolder_MD,'*.nii.gz'));
dirOutput_concussion_MD=dir(fullfile(fileFolder_concussion_MD,'*.nii.gz'));

% AD modality
dirOutput_AD=dir(fullfile(fileFolder_AD,'*.nii.gz'));
dirOutput_concussion_AD=dir(fullfile(fileFolder_concussion_AD,'*.nii.gz'));

% RD modality
dirOutput_RD=dir(fullfile(fileFolder_RD,'*.nii.gz'));
dirOutput_concussion_RD=dir(fullfile(fileFolder_concussion_RD,'*.nii.gz'));

% number of cases
num_case=length(dirOutput_concussion_FA)+length(dirOutput_FA);

% initialize parameters
FA_data=zeros(num_case,length(index));
MD_data=zeros(num_case,length(index));
AD_data=zeros(num_case,length(index));
RD_data=zeros(num_case,length(index));

label=zeros(num_case,1);
% Obtain file names
fileNames_FA={dirOutput_FA.name}';
fileNames_concussion_FA={dirOutput_concussion_FA.name}';

fileNames_MD={dirOutput_MD.name}';
fileNames_concussion_MD={dirOutput_concussion_MD.name}';

fileNames_AD={dirOutput_AD.name}';
fileNames_concussion_AD={dirOutput_concussion_AD.name}';

fileNames_RD={dirOutput_RD.name}';
fileNames_concussion_RD={dirOutput_concussion_RD.name}';
%% Assign label for non-concussion cases
for i=1:length(fileNames_FA)
    im_FA=readFileNifti([fileFolder_FA,fileNames_FA{i}]);   
    im_MD=readFileNifti([fileFolder_MD,fileNames_MD{i}]);
    im_AD=readFileNifti([fileFolder_AD,fileNames_AD{i}]);
    im_RD=readFileNifti([fileFolder_RD,fileNames_RD{i}]);
    
    FA_data(i,:)=im_FA.data(index);
    MD_data(i,:)=im_MD.data(index);
    AD_data(i,:)=im_AD.data(index);
    RD_data(i,:)=im_RD.data(index);
    label(i)=0;
    i
end

%% Assign label for concussion cases
for i=1:length(fileNames_concussion_FA)

    im_con_FA=readFileNifti([fileFolder_concussion_FA,fileNames_concussion_FA{i}]);    
    im_con_MD=readFileNifti([fileFolder_concussion_MD,fileNames_concussion_MD{i}]);
    im_con_AD=readFileNifti([fileFolder_concussion_AD,fileNames_concussion_AD{i}]);
    im_con_RD=readFileNifti([fileFolder_concussion_RD,fileNames_concussion_RD{i}]);
    
    FA_data(i+length(fileNames_FA),:)=im_con_FA.data(index);
    MD_data(i+length(fileNames_FA),:)=im_con_MD.data(index);
    AD_data(i+length(fileNames_FA),:)=im_con_AD.data(index);
    RD_data(i+length(fileNames_FA),:)=im_con_RD.data(index);
    label(i+length(fileNames_FA))=1;
    i
end
%% save file
save('genctrl_mildTBI_TBSS.mat','FA_data','MD_data','AD_data','RD_data','label');
