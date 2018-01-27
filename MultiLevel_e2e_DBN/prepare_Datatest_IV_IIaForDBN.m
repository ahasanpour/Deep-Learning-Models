 function [ perc,kappa ] = prepare_Datatest_IV_IIaForDBN()


pathTrain='/home/ahp/MyMatlabProjects/IV_2a/A1Th.mat';
pathTest='/home/ahp/MyMatlabProjects/IV_2a/A1Eh.mat';

%----------------------------------separate train
load(pathTrain);
% channels=[8,9,10,11,12];
channels=[8,12];
%channels=10;
finishTime=35;
Fs = 250;            % Sampling frequency
T = 1/Fs;             % Sampling period
L = 546;             % Length of signal
jj=1;
sig=[];
for kk=1:288
    if h.Classlabel(kk,1)==1 || h.Classlabel(kk,1)==2 || h.Classlabel(kk,1)==3 || h.Classlabel(kk,1)==4
        if length(sig)==0
           tri(jj,1)=1;
       else
            tri(jj,1)=length(sig)+1;
        end
       
            sig=[sig ; s(h.TRIG(kk,1):h.TRIG(kk,1)+1749,channels)];
            Labels1(jj,1)=h.Classlabel(kk,1);jj=jj+1;
        
    end
    
end

signals=sig;
trigs=tri;
Labels=Labels1; 

%--------------------------------------separate test
load(pathTest);

jj=1;
sigt=[];
for kk=1:288
    if h.Classlabel(kk,1)==1 || h.Classlabel(kk,1)==2 || h.Classlabel(kk,1)==3 || h.Classlabel(kk,1)==4
        if length(sigt)==0
           trit(jj,1)=1;
       else
            trit(jj,1)=length(sigt)+1;
        end
       
            sigt=[sigt ; s(h.TRIG(kk,1):h.TRIG(kk,1)+1749,channels)];
            Labels1t(jj,1)=h.Classlabel(kk,1);jj=jj+1;
        
    end
    
end

signalst=sigt;
trigst=trit;
Labelst=Labels1t; 
%---------------------------
startPoint=580;
endpoint=1125;
m=2;
[One Two Three Four]=separateClass(signals ,Labels, trigs);

j=1;    
        for h=1:1750:length(One)
          
            %Cov1=(One(h+startPoint:h+endpoint,:))'; 
            %featureClass1(j,:)=(Cov1(:))';
            Cov1=(One(h+startPoint:h+endpoint,:))';
            for uu=1:size(Cov1,1)
                Y(uu,:) = (fft(Cov1(uu,:)))/L;
                PP1 = Y(uu,1:L/2+1);
                PPP1(uu,:) = 2*PP1(1,8:finishTime-1);
                
            end
            featureClass1(j,:)=real(reshape(PPP1',size(PPP1,1)*size(PPP1,2),1))';
            %subplot(2,1,j);
            %plot(featureClass1(j,1:70));
            
            j=j+1;
        end
        j=1;
        for h=1:1750:length(Two)
            
            Cov1=(Two(h+startPoint:h+endpoint,:))';

            for uu=1:size(Cov1,1)
                Y(uu,:) = (fft(Cov1(uu,:)))/L;
                PP1 = Y(uu,1:L/2+1);
                PPP1(uu,:) = 2*PP1(1,8:finishTime-1);
                
            end
            featureClass2(j,:)=real(reshape(PPP1',size(PPP1,1)*size(PPP1,2),1))';
            j=j+1;
        end
        j=1;
        for h=1:1750:length(Three)
            
            Cov1=(Three(h+startPoint:h+endpoint,:))';

            for uu=1:size(Cov1,1)
                Y(uu,:) = (fft(Cov1(uu,:)))/L;
                PP1 = Y(uu,1:L/2+1);
                PPP1(uu,:) = 2*PP1(1,8:finishTime-1);
                
            end
            featureClass3(j,:)=real(reshape(PPP1',size(PPP1,1)*size(PPP1,2),1))';
            j=j+1;
        end
        j=1;
        for h=1:1750:length(Four)
            
            Cov1=(Four(h+startPoint:h+endpoint,:))';

            for uu=1:size(Cov1,1)
                Y(uu,:) = (fft(Cov1(uu,:)))/L;
                PP1 = Y(uu,1:L/2+1);
                PPP1(uu,:) = 2*PP1(1,8:finishTime-1);
                
            end
            featureClass4(j,:)=real(reshape(PPP1',size(PPP1,1)*size(PPP1,2),1))';
            j=j+1;
        end
        featureTrain=[featureClass1 ;featureClass2 ];%featureClass3 ;featureClass4];
        trainTarget(1:72,1)=ones(72,1);
        trainTarget(73:144,1)=ones(72,1)*2;
%         trainTarget(145:216,1)=ones(72,1)*3;
%         trainTarget(217:288,1)=ones(72,1)*4;
        
        %----------------------- separate test
        
        [One Two Three Four]=separateClass(signalst ,Labelst, trigst);

j=1;    
        for h=1:1750:length(One)
          
            Cov1=(One(h+startPoint:h+endpoint,:))';

            for uu=1:size(Cov1,1)
                Y(uu,:) = (fft(Cov1(uu,:)))/L;
                PP1 = Y(uu,1:L/2+1);
                PPP1(uu,:) = 2*PP1(1,8:finishTime-1);
                
            end
            featureClass1(j,:)=real(reshape(PPP1',size(PPP1,1)*size(PPP1,2),1))';
           
            j=j+1;
        end
        j=1;
        for h=1:1750:length(Two)
            
            Cov1=(Two(h+startPoint:h+endpoint,:))';

            for uu=1:size(Cov1,1)
                Y(uu,:) = (fft(Cov1(uu,:)))/L;
                PP1 = Y(uu,1:L/2+1);
                PPP1(uu,:) = 2*PP1(1,8:finishTime-1);
                
            end
            featureClass2(j,:)=real(reshape(PPP1',size(PPP1,1)*size(PPP1,2),1))';
            j=j+1;
        end
        j=1;
        for h=1:1750:length(Three)
            
            Cov1=(Three(h+startPoint:h+endpoint,:))';

            for uu=1:size(Cov1,1)
                Y(uu,:) = (fft(Cov1(uu,:)))/L;
                PP1 = Y(uu,1:L/2+1);
                PPP1(uu,:) = 2*PP1(1,8:finishTime-1);
                
            end
            featureClass3(j,:)=real(reshape(PPP1',size(PPP1,1)*size(PPP1,2),1))';
            j=j+1;
        end
        j=1;
        for h=1:1750:length(Four)
            
            Cov1=(Four(h+startPoint:h+endpoint,:))';

            for uu=1:size(Cov1,1)
                Y(uu,:) = (fft(Cov1(uu,:)))/L;
                PP1 = Y(uu,1:L/2+1);
                PPP1(uu,:) = 2*PP1(1,8:finishTime-1);
                
            end
            featureClass4(j,:)=real(reshape(PPP1',size(PPP1,1)*size(PPP1,2),1))';
            j=j+1;
        end
        featureTest=[featureClass1 ;featureClass2 ];%featureClass3 ;featureClass4];
        TestTarget(1:72,1)=ones(72,1);
        TestTarget(73:144,1)=ones(72,1)*2;
%         TestTarget(145:216,1)=ones(72,1)*3;
%         TestTarget(217:288,1)=ones(72,1)*4;
        
        [dataTr,targetTrain,dataTe,targetTest]=shuffleM(featureTrain,trainTarget,featureTest,TestTarget);
        dataTrain = MeanNormalize (dataTr);
        
        dataTest = MeanNormalize (dataTe);
        %dataTest=OneOrZero(dataTest)
        %dataTrain=OneOrZero(dataTrain)
        dataTrain=[ones(1,size(dataTrain,2));dataTrain];
        dataTest=[ones(1,size(dataTest,2));dataTest];
        targetTrain=[1;targetTrain];
        targetTest=[1;targetTest];
        csvwrite('/home/ahp/MyMatlabProjects/IV_2a/subjecta/dataTrainfftTwoChannelsTwoClass.csv', dataTrain);
        csvwrite('/home/ahp/MyMatlabProjects/IV_2a/subjecta/dataTestfftTwoChannelsTwoClass.csv', dataTest);
        csvwrite('/home/ahp/MyMatlabProjects/IV_2a/subjecta.csv', targetTrain);
        csvwrite('/home/ahp/MyMatlabProjects/IV_2a/subjecta/targetTest.csv', targetTest);
i=1;
end
function [one two three four]=separateClass(data,label,trig)
    one=[];
    two=[];
    three=[];
    four=[];
    for r=1:length(trig)
       if label(r,1)==1
           one=[one;data(trig(r,1):trig(r,1)+1749,:)];
       elseif label(r,1)==2
           two=[two;data(trig(r,1):trig(r,1)+1749,:)];
       elseif label(r,1)==3
            three=[three;data(trig(r,1):trig(r,1)+1749,:)];   
       elseif label(r,1)==4
           four=[four;data(trig(r,1):trig(r,1)+1749,:)];
       end
    end
        
end
