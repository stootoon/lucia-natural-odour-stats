
%import excel table (with odours as rows, so the names can be used later,
%otherwise cannot access the odourant names) call it tablet. 
data= readtable(['auer-for-matlab.csv']); % import data from excel call it data
tablet= data(:, 1:end);

%%
    x= tablet{:,2:end}'; % this is to convert from table to matrix, and also exclude the column with odour names

%%

%%thresh_log= x>10000; % this will give me logical array for x bigger than 10^5 (this value is for pandan samples analysed with amdis, for masshunter library output- 10^4 seems for reaasonable- later used  50000.
%%x_thresh= x.*thresh_log; % make the threshold of the data- 

% no thresholding for dweck fruit data since they are proportions
%x_thresh= x;
%%
zscor_xnan = @(x) bsxfun(@rdivide, bsxfun(@minus, x, mean(x,'omitnan')), std(x, 'omitnan'));

xx= zscor_xnan(x')'; % does the following steps in one line basically does the zscore and puts in fruits vs odours table for pca
%a = x';
%aa = zscor_xnan (a);
%xx = aa';
%xx2= zscor_xnan(x2')'; 
[coeff,score,latent,tsquared,explained,mu] = pca(xx); % principal component analysis for n by p data matrix. Rows (n) corresponds to observations and columns (p) correspond to variables.
                 % coeff matrix is p by p. Each coumn of coeff contains
                 % coefficients for one pincipal component and the columns
                 % are in descending order of component variance. PCA
                 % default uses singular value decomposition algorithm. 




%% prepare variance explained

exp = explained';

coeffn= zeros(size(coeff));



for i = 1:size(coeff,2) 
    
coeffn(:,i) =  coeff (:,i)*exp(i);
end


%% ---------- PCscore (absolute variance-weighted loadings, first 3 PCs) ----------
% explained is already percent variance explained (like 17.5, 9.06, 8.35...)
w = explained(1:3);                 % 3x1

% PCscore per odour (variables = rows of coeff)
% PCscore = sum_k |coeff(:,k)| * explained(k)
PCscore = sum( abs(coeff(:,1:3)) .* w.', 2 );   % odours x 1

%% ---------- Make a table with odour names + PCscore ----------
odourNames = string(tablet{:,1});  % first column of your input file

PCscoreTable = table(odourNames, PCscore, ...
    'VariableNames', {'Odour','PC_score_rowZ'});

PCscoreTable.Odour = strtrim(PCscoreTable.Odour);

%% ---------- Load behaviour annotations and merge ----------
ann = readtable('behaviour-anotations.xlsx');

% Make sure odour column is named 'Odour'
if ~any(strcmpi(ann.Properties.VariableNames, 'Odour'))
    error("Annotation file must contain a column named 'Odour'. Current columns: %s", ...
          strjoin(string(ann.Properties.VariableNames), ", "));
end

ann.Odour = strtrim(string(ann.Odour));

M = innerjoin(ann, PCscoreTable, 'Keys', 'Odour');

% Ignore aversive (-1)
M = M(M.behaviour ~= -1, :);

%% ---------- Labels ----------
behLabel = strings(height(M),1);
behLabel(M.behaviour == 1)   = "Attractive";
behLabel(M.behaviour == 0)   = "Neutral";
behLabel(M.behaviour == 0.5) = "Conflicting";

keep = behLabel ~= "";
M = M(keep,:);
behLabel = categorical(behLabel(keep), ["Attractive","Neutral","Conflicting"]);

%% ---------- Boxplot + individual datapoints ----------
figure('Color','w');

% Draw boxplot
boxplot(M.PC_score_rowZ, behLabel, 'Symbol',''); % removes default outlier stars
hold on

% Convert categories to numeric positions
cats = categories(behLabel);
xpos = grp2idx(behLabel);   % 1,2,3 positions

% Add horizontal jitter so points don't overlap
jitterAmount = 0.18;
xjitter = xpos + (rand(size(xpos)) - 0.5) * 2 * jitterAmount;

% Plot datapoints
scatter(xjitter, M.PC_score_rowZ, ...
        35, ...            % marker size
        'o', ...
        'filled', ...
        'MarkerFaceAlpha', 0.6, ...
        'MarkerEdgeColor','k');

% Axis labels
ylabel('Row-zscore absolute PC score (PC1–PC3)');
title('Behaviour vs Row-zscore PC score');

% Fix y-axis range
ylim([0 19]);

% Make nicer
box on
set(gca,'FontSize',12)
hold off


%% plot graph of variance explained- cumulative plot

n= size(coeff,2);
%n2= size(coeff2,2);

m = 1:n;
%m2= 1:n2;


figure
%subplot(2,2,1)
h = plot (cumsum(explained),'.b');
set(h,'MarkerSize',30);
set(h,'Clipping', 'off');
set(gcf,'color','w')
set(gca,'XTick', 1:n,'XTickLabel',m)
xlim([1,10]) 
ylim([0,100]);
set(gca,'TickDir','out')
set(gca,'FontSize',16)
set(gca,'box','off')
xlabel('PC')
ylabel('Cumulative variance explained')
%% PCA loadings * variance explained
figure
k = stem(coeffn(:,1:size(coeffn,2))','.');
set(k,'MarkerSize', 30);
set(gcf,'color','w')
set(gca,'XTick', 1:4,'XTickLabel',m)
xlim([0.2,4.2])
%ylim([-40, 40])
%ylim([-20,30])
set(gca,'TickDir','out')
set(gca,'FontSize',16)
set(gca,'box','off')
%legend ('show')
hold on
plot([0,100],[ 0 0], 'k-')
hold off
xlabel('PC')
ylabel('Loading*variance explained')
%title('PCA peak Amp Dmel Dere V-avg')


%% finding the odours for loadings only- the result is exactly the same, since *variance is only change in magnitude
% change coeff to coeffn if need the loadings*variance explained numbers in
% odcoeff
% orders by descending order, and then pick out the first and second max
% and minimum for the first 3 pcs. 


odours= tablet{:,1}; % obtain the odours names from original table of data 
odcoeff= table(odours, coeff(:,1), coeff(:,2), coeff(:,3), coeff(:,4)); % create table with odour names on the left side of the coeffs of each PC represented on the plot
%using coeff here instead of coeffn, but it's the same, since multipliying
%by a scalar.

% PC1
odmax_pc1= sortrows(odcoeff,2,'descend'); % pc1 sort min to max , 'descend' makes from max to min
first_pc1= odmax_pc1{[1 end],1}; % gets the minimum and maximum values of pc1 (the odours accounting for highest variance explaine)
second_pc1 = odmax_pc1{[2, end-1],1}; % gets SECOND min and max of pc1 (odours accounting second highest variance 
pc1= [first_pc1, second_pc1]

% PC2
odmax_pc2= sortrows(odcoeff,3, 'descend'); % same as for PC1
first_pc2= odmax_pc2{[1 end],1};
second_pc2= odmax_pc2{[2 end-1],1};


pc2= [first_pc2, second_pc2]  % min and max in second pc. 

% PC3 
odmax_pc3= sortrows(odcoeff, 4, 'descend');
first_pc3= odmax_pc3{[1 end],1};
second_pc3= odmax_pc3{[2, end-1],1};

pc3= [first_pc3, second_pc3]

                 
%% PC1, PC2 figure 

figure

plot(score(1:4,1), score(1:4,2),'.r','markersize',20)%S10
hold on
plot(score(5:8,1), score(5:8,2),'.','markersize',20, 'Color', [0.6, 0.1,0.3]) % S11

plot(score(9:13,1), score(9:13,2),'.g','markersize',20) % S11a

plot(score(14:18,1), score(14:18,2),'.b','markersize',20) %s12

plot(score(19:22,1), score(19:22,2),'.m','markersize',20) % S14

plot(score(23:26,1), score(23:26,2),'.','markersize',20,'Color', [0.8, 0.8,0]) % S15

plot(score(27:31,1), score(27:31,2),'.c','markersize',20) % S9
hold off




xlabel('1st Principal Component')
ylabel('2nd Principal Component')

legend('S10', 'S11', 'S11a', 'S12', 'S14', 'S15', 'S9', 'Location', 'bestoutside') 
%% PC1, PC2, PC3
figure
plot3(score(27:31,1), score(27:31,2), score(27:31,3),'.c','markersize',20) % S9
hold on 
plot3(score(1:4,1), score(1:4,2),score(1:4,3),'.r','markersize',20)%S10

plot3(score(5:8,1), score(5:8,2),score(5:8,3),'.','markersize',20, 'Color', [0.6, 0,0.3]) % S11

plot3(score(9:13,1), score(9:13,2),score(9:13,3),'.g','markersize',20) % S11a

plot3(score(14:18,1), score(14:18,2), score(14:18,3),'.b','markersize',20) %s12

plot3(score(19:22,1), score(19:22,2), score(19:22,3),'.m','markersize',20) % S14

plot3(score(23:26,1), score(23:26,2), score(23:26,3),'.','markersize',20,'Color', [0.8, 0.8,0.8]) % S15
hold off


hold off
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')

grid on
legend('S9','S10', 'S11', 'S11a', 'S12', 'S14', 'S15' , 'Location', 'bestoutside') 

