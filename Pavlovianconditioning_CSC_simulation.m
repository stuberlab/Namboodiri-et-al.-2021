%% Pavlovian Conditioning: Set up state space

% States: 

% ITI states
% Split maximum duration of ITI into equally long states of duration
% statedur.
% Mean ITI = 30 s => probability of transitioning to cue
% from any ITI state is ITI state duration/30s

% CS+ to reward delay states
% CS- to omission delay states

%% Set up CSC state space

rng(7); % Set seed to a defined value

maxITI = 90; 
maxcuerewdelay = 3;
outcomedelay = 3;

% Duration of a single state within the ITI, cue or reward
statedur = 3;

numITIstates = maxITI/statedur;
numstatescsplus = (maxcuerewdelay+outcomedelay)/statedur;
numstatescsminus = (maxcuerewdelay+outcomedelay)/statedur;
statespacesize = numITIstates + numstatescsplus + numstatescsminus;

% Index for each state type in the state space
csplusstates = numITIstates+1:numITIstates + numstatescsplus;
csminusstates = numITIstates+numstatescsplus+1:numITIstates + numstatescsplus + numstatescsminus;
ITIstates = 1:numITIstates;
%% Simulate experiment: Initial learning at full contingency

csplus_reward_prob = 1;
meanITI = 30; % in seconds
numcsplus = 5000;
numcsminus = 5000;
backgroundrewperiod = NaN;

[cuetimes, csplusflag, rewardtimes] = ...
simulateexperiment(csplus_reward_prob, meanITI, maxITI,...
                   numcsplus, numcsminus, maxcuerewdelay, outcomedelay,...
                   backgroundrewperiod);
% Make sure minimum cue time is at least one second from experiment start               
cuetimes = cuetimes + 1;      
rewardtimes = rewardtimes + 1;
sessionendtime = cuetimes(end) + maxcuerewdelay + outcomedelay;

%% Contingency degradation experiment

contingencydegradation = '50percent';
% contingencydegradation = 'background';

if strcmp(contingencydegradation, '50percent')
    %Reduce reward probability
    csplus_reward_prob = 0.5;
    % numcsplus = 5000;
    % numcsminus = 5000;
    % numcues = numcsplus + numcsminus;
    backgroundrewperiod = NaN;

    [cuetimes1, csplusflag1, rewardtimes1] = ...
    simulateexperiment(csplus_reward_prob, meanITI, maxITI,...
                       numcsplus, numcsminus, maxcuerewdelay, outcomedelay,...
                       backgroundrewperiod);

    cuetimes = [cuetimes;sessionendtime+cuetimes1];
    csplusflag = [csplusflag;csplusflag1];
    rewardtimes = [rewardtimes;sessionendtime+rewardtimes1];
elseif strcmp(contingencydegradation, 'background')
    % Background reward rate
    csplus_reward_prob = 1;
    % numcsplus = 5000;
    % numcsminus = 5000;
    % numcues = numcsplus + numcsminus;
    backgroundrewperiod = 30;

    [cuetimes1, csplusflag1, rewardtimes1] = ...
    simulateexperiment(csplus_reward_prob, meanITI, maxITI,...
                       numcsplus, numcsminus, maxcuerewdelay, outcomedelay,...
                       backgroundrewperiod);

    cuetimes = [cuetimes;sessionendtime+cuetimes1];
    csplusflag = [csplusflag;csplusflag1];
    rewardtimes = [rewardtimes;sessionendtime+rewardtimes1];
end
%% Set up state timeline

numstatesinsession = floor((cuetimes(end) + maxcuerewdelay + outcomedelay)/statedur)-1;
statetimeline = NaN(numstatesinsession, 1);

statenumberforcuetimes = floor(cuetimes/statedur);


temp = 1;
i = 1;
while i<=length(statenumberforcuetimes)
    temp1 = statenumberforcuetimes(i)-temp;
    statetimeline(temp:temp+temp1-1) = ITIstates(1:temp1);  
    if csplusflag(i)==1
        statetimeline(temp+temp1:temp+temp1+numstatescsplus-1) = csplusstates;  
        temp = statenumberforcuetimes(i) + numstatescsplus;
    elseif csplusflag(i)==0
        statetimeline(temp+temp1:temp+temp1+numstatescsminus-1) = csminusstates;  
        temp = statenumberforcuetimes(i) + numstatescsminus;
    end
    i = i+1;
end

rewardtimeline = zeros(numstatesinsession, 1);

temp = floor(rewardtimes/statedur);
rewardtimeline(temp) = 1;

%% Learn value
alphainit = 0.1;

% set_learning_rate = 'constant';
set_learning_rate = 'metalearning';

alpha = 0.1;
rewardmag = 1;
gamma = 0.99;

window = 5000;
eta = 1;
kappa = 5;

if strcmp(set_learning_rate, 'constant')
    deltaalpha = 0.02; % for constant learning rate (needs more noise for
    % variability)
elseif strcmp(set_learning_rate, 'metalearning')
    deltaalpha = 0.01; % for metalearning
end

valuesforstate = zeros(1, statespacesize);
RPEtimeline = NaN(numstatesinsession, 1);
alphatimeline = NaN(numstatesinsession, 1);
volatilitytimeline = NaN(numstatesinsession, 1);
zscoretimeline = NaN(numstatesinsession, 1);


alpha_trials = NaN(sum(csplusflag),1);
value_trials = NaN(sum(csplusflag),1);
RPE_trials = NaN(sum(csplusflag),1);

temp = 1;
for i = 1:numstatesinsession-1
    
    if strcmp(set_learning_rate, 'metalearning')
        % Alpha controlled by volatility
        if i>window
            meanRPEwindow = nanmean(RPEtimeline(i-window:i-1));
            stdRPEwindow = nanstd(RPEtimeline(i-window:i-1));
            zscoretimeline(i) = (RPE-meanRPEwindow)/stdRPEwindow;
            if i>2*window
                abszscore = abs(nanmean(zscoretimeline(i-window:i)));
                volatility = kappa*abszscore/(1+kappa*abszscore);
                alpha = alpha + eta*(volatility-alpha)+deltaalpha*randn;
                volatilitytimeline(i) = volatility;
            end
        end
    elseif strcmp(set_learning_rate, 'constant')  
        % Flat alpha plus noise
        alpha = alphainit + deltaalpha*randn;
    end

    if alpha < 0
        alpha = 0;
    end
    
    RPE = rewardtimeline(i+1) + gamma*valuesforstate(statetimeline(i+1))-...
          valuesforstate(statetimeline(i));
    RPEtimeline(i) = RPE;
    
    alphatimeline(i) = alpha;
    if statetimeline(i)==csplusstates(1)
        alpha_trials(temp) = alpha;
        value_trials(temp) = valuesforstate(statetimeline(i));
        RPE_trials(temp) = RPE;
        temp = temp+1;
    end
    valuesforstate(statetimeline(i))=valuesforstate(statetimeline(i))+alpha*RPE;
end
%% Plot learning rate
figure1 = figure;
set(figure1,'Position',[10 10 200 200])
set(figure1,'color','w');

yyaxis left
plot(-1000:1000, value_trials(numcsplus-1000:numcsplus+1000));
ylabel('CS+ value');
yyaxis right
plot(-1000:1000, alpha_trials(numcsplus-1000:numcsplus+1000));

xlabel({'Trial number from';'contingency degradation'});
ylabel('Learning rate');

%% Define mapping between value and behavior

behavior_mapping = 'linear';
% behavior_mapping = 'nonlinear';

if strcmp(behavior_mapping, 'linear')
    behavior_trials = value_trials; % Linear mapping
elseif strcmp(behavior_mapping, 'nonlinear')
    lickmax = 10;  %Max lick rate
    halfpoint = 3;
    tau = 0.4;
    behavior_trials = lickmax./(1+exp(-(value_trials-halfpoint)/tau));
    
    plot(value_trials, behavior_trials);
    xlabel('Cue value');
    ylabel('Anticipatory licking');
    set(gcf,'Position',[10 10 200 200])
    set(gcf,'color','w');
end

%% Plot Learning rate vs value update

% Plot results for phase 2 after contingency degradation

numtrialsphase1 = length(cuetimes(csplusflag==1))-length(cuetimes1(csplusflag1==1));
tempstart = numtrialsphase1 + 1;
% tempstart = length(alpha_trials)-299;
% tempend = numtrialsphase1 + 300;
tempend = length(alpha_trials);

figure1 = figure;
set(figure1,'Position',[10 10 200 200])
set(figure1,'color','w');

figure2 = figure;
set(figure2,'Position',[10 10 200 200])
set(figure2,'color','w');

figure(figure1);
tempalpha = alpha_trials(tempstart:tempend-1);
tempRPE = RPE_trials(tempstart:tempend-1);
tempvalue = value_trials(tempstart:tempend-1);
tempbehaviorupdate = diff(behavior_trials(tempstart:tempend));
tempx = tempalpha(tempRPE>0);
% tempx = tempRPE(tempRPE>0);

tempy = tempbehaviorupdate(tempRPE>0);
ztempx = (tempx-mean(tempx))/std(tempx);
ztempy = (tempy-mean(tempy))/std(tempy);
scatter(ztempx, ztempy, 'b'); hold on

% [rho, pval] = corr(ztempx, ztempy)

figure(figure2);
bins = [-3, -1, 1, 3, 5];
ztempxbinned = discretize(ztempx, bins);
for i = 1:length(bins)
    temp = ztempy(ztempxbinned==i);
    scatter(bins(i), mean(temp), 'b', 'filled');hold on
    errorbar(bins(i), mean(temp), std(temp)/sqrt(length(temp)),...
        'LineStyle','none', 'Color','b', 'CapSize',0, 'LineWidth', 1.5);
    hold on
end



figure(figure1);
tempx = tempalpha(tempRPE<0);
% tempx = tempRPE(tempRPE<0);
tempy = tempbehaviorupdate(tempRPE<0);
ztempx = (tempx-mean(tempx))/std(tempx);
ztempy = (tempy-mean(tempy))/std(tempy);
scatter(ztempx, ztempy, 'r'); hold on
xlim([-5, 5]);ylim([-5, 5]);
xticks([-4, 0, 4]);yticks([-4, 0, 4]);


% [rho, pval] = corr(ztempx, ztempy)

figure(figure2);
bins = [-3, -1, 1, 3, 5];
ztempxbinned = discretize(ztempx, bins);
for i = 1:length(bins)
    temp = ztempy(ztempxbinned==i);
    scatter(bins(i), mean(temp), 'r', 'filled');hold on
    errorbar(bins(i), mean(temp), std(temp)/sqrt(length(temp)),...
        'LineStyle','none', 'Color','r', 'CapSize',0, 'LineWidth', 1.5);
    hold on
end
xlim([-4, 4]);ylim([-5, 5]);
xticks([-3, -1, 1, 3]);yticks([-3, 0, 3]);

%%
function [cuetimes, csplusflag, rewardtimes] =...
    simulateexperiment(csplus_reward_prob, meanITI, maxITI,...
                       numcsplus, numcsminus, maxcuerewdelay, outcomedelay,...
                       backgroundrewperiod)
                   
    numcues = numcsplus + numcsminus;
    cuetimes = NaN(numcues,1);
    csplusflag = [ones(numcsplus,1); zeros(numcsminus,1)];
    csplusflag = csplusflag(randperm(length(csplusflag)));
    rewardflag = zeros(numcues,1);
    temp = 0;
    if isfinite(backgroundrewperiod)
        tempbgdrewards = NaN(ceil(3*(maxITI*numcues)/backgroundrewperiod),1);
        tempbgdrewidx = 1;
    end
    for i = 1:numcues
        cuetimes(i) = temp + min(exprnd(meanITI), maxITI);
        if isfinite(backgroundrewperiod)
            tempbgd = temp + cumsum(exprnd(backgroundrewperiod, ceil(3*maxITI/backgroundrewperiod)));
            tempbgd = tempbgd(tempbgd<cuetimes(i));
            tempbgdrewards(tempbgdrewidx:tempbgdrewidx+length(tempbgd)-1) = tempbgd;
            tempbgdrewidx = tempbgdrewidx + length(tempbgd);
        end
        temp = cuetimes(i) + maxcuerewdelay + outcomedelay;
        if csplusflag(i) == 1
            if rand<csplus_reward_prob
                rewardflag(i) = 1;
            end
        end
    end
    rewardtimes = cuetimes(rewardflag==1)+maxcuerewdelay;
    if isfinite(backgroundrewperiod)
        tempbgdrewards = tempbgdrewards(isfinite(tempbgdrewards));
        rewardtimes = sort([rewardtimes;tempbgdrewards]);
    end
end