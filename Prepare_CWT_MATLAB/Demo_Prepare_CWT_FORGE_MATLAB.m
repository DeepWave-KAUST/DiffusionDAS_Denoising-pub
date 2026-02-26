clc;clear;close all;

% Loading Noisy DAS Data
% eq = 74;
timesteps = 2000;
load('out_eq_74.mat')
clip=99;


tic()

% BandPass Filter
dt = 0.0005; % Sampling
outF=das_bandpass(dn,dt,0,250,6,6,0,0); %Setting Low and High cutting Freqs to Remove hHgh-Freq Noise.

% CWT Scales
wtmexh1 = cwtft2(outF,'wavelet','mexh','scales',1:0.5:10);
wtmexh = cwtft2(outF,'wavelet','paul','scales',1:0.5:10);

s=size(wtmexh.cfs);
outr = real(wtmexh.cfs(:,:,1,s(4))); % Real PAUL
outi = imag(wtmexh.cfs(:,:,1,s(4))); % Imag PAUL
out = wtmexh1.cfs(:,:,1,s(4)); % MexHat

toc()


% Plotting
samples = 1:length(out(:,1));         % Sample Indices Vector
Fs = 1/dt;             % Sampling Frequency (Hz)
t = samples/Fs;         % Time Vector (seconds)
ax = 1:length(out(1,:));
ay = [0,t]; 

figure(2);das_imagesc(outr,clip,1,ax,ay)
xlabel('Channel','FontSize',10,'FontWeight','bold')
ylabel('Time (s)','FontSize',10,'FontWeight','bold')
title('CWT (real PAUL)','FontSize',10,'FontWeight','bold')
set(gca,'Linewidth',1,'FontSize',10,'Fontweight','bold');
%print(gcf, ['Final_Paper_DAS_Figures/Noisy_',num2str(eq),'.jpeg'],'-djpeg','-r300');

figure(222);das_imagesc(outi,clip,1,ax,ay)
xlabel('Channel','FontSize',10,'FontWeight','bold')
ylabel('Time (s)','FontSize',10,'FontWeight','bold')
title('CWT (Imag PAUL)','FontSize',10,'FontWeight','bold')
set(gca,'Linewidth',1,'FontSize',10,'Fontweight','bold');
%print(gcf, ['Final_Paper_DAS_Figures/Noisy_',num2str(eq),'.jpeg'],'-djpeg','-r300');

figure(200);das_imagesc(out,clip,1,ax,ay)
xlabel('Channel','FontSize',10,'FontWeight','bold')
ylabel('Time (s)','FontSize',10,'FontWeight','bold')
title('CWT (MexHat)','FontSize',10,'FontWeight','bold')
set(gca,'Linewidth',1,'FontSize',10,'Fontweight','bold');
%print(gcf, ['Final_Paper_DAS_Figures/Noisy_',num2str(eq),'.jpeg'],'-djpeg','-r300');



figure(1);das_imagesc(dn,clip,1,ax,ay)
xlabel('Channel','FontSize',10,'FontWeight','bold')
ylabel('Time (s)','FontSize',10,'FontWeight','bold')
title('Noisy','FontSize',10,'FontWeight','bold')
set(gca,'Linewidth',1,'FontSize',10,'Fontweight','bold');
%print(gcf, ['Final_Paper_DAS_Figures/Noisy_',num2str(eq),'.jpeg'],'-djpeg','-r300');



figure(5);das_imagesc(outF,clip,1,ax,ay)
xlabel('Channel','FontSize',10,'FontWeight','bold')
ylabel('Time (s)','FontSize',10,'FontWeight','bold')
title('BP','FontSize',10,'FontWeight','bold')
set(gca,'Linewidth',1,'FontSize',10,'Fontweight','bold');
%print(gcf, ['Final_Paper_DAS_Figures/Noisy_',num2str(eq),'.jpeg'],'-djpeg','-r300');

figure(6);das_imagesc(outF-dn,clip,1,ax,ay)
xlabel('Channel','FontSize',10,'FontWeight','bold')
ylabel('Time (s)','FontSize',10,'FontWeight','bold')
title('BP-Diff','FontSize',10,'FontWeight','bold')
set(gca,'Linewidth',1,'FontSize',10,'Fontweight','bold');
%print(gcf, ['Final_Paper_DAS_Figures/Noisy_',num2str(eq),'.jpeg'],'-djpeg','-r300');

% Saving
save(['out_Diff_',num2str(eq),'_pre','.mat'],'dn','outi','outr','outF','out')
