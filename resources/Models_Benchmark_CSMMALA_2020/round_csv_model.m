function round_csv_model(basePath, modelName) 
options.toRound = 1;
options.fullDim = 1;


fileA = strcat(basePath, '/A_', modelName, '_unrounded.csv')
fileB = strcat(basePath, '/b_', modelName, '_unrounded.csv')


model = {};

model.A = csvread(fileA, 0, 0);
model.b = csvread(fileB, 0, 0);

center = getCCcenter(model.A, model.b);

roundedModel = preprocess(model, options);


roundedCenter = getCCcenter(roundedModel.A, roundedModel.b);


dlmwrite(strcat(basePath, '/A_', modelName, '_rounded.csv'), roundedModel.A, 'delimiter', ',', 'precision', 15);
dlmwrite(strcat(basePath, '/b_', modelName, '_rounded.csv'), roundedModel.b, 'delimiter', ',', 'precision', 15);
dlmwrite(strcat(basePath, '/T_', modelName, '_rounded.csv'), roundedModel.T, 'delimiter', ',', 'precision', 15);
dlmwrite(strcat(basePath, '/N_', modelName, '_rounded.csv'), roundedModel.N, 'delimiter', ',', 'precision', 15);
dlmwrite(strcat(basePath, '/p_shift_', modelName, '_rounded.csv'), roundedModel.p_shift, 'delimiter', ',', 'precision', 15); 
dlmwrite(strcat(basePath, '/start_', modelName, '_rounded.csv'), roundedCenter, 'delimiter', ',', 'precision', 15);
dlmwrite(strcat(basePath, '/start_', modelName, '.csv'), center, 'delimiter', ',', 'precision', 15);



%compute the center of the Chebyshev ball in the polytope Ax<=b
function [CC_center,radius] = getCCcenter(A,b)

dim = size(A,2);
a_norms = sqrt(sum(A.^2,2));

LP.A = [A a_norms];
LP.b = b;
LP.c = [zeros(dim,1); 1];
LP.lb = -Inf * ones(dim+1,1);
LP.ub = Inf*ones(dim+1,1);
LP.osense = -1;
LP.csense = repmat('L',size(LP.A,1),1);
solution = solveCobraLP(LP);

%also allow -1 because it might be good enough
if solution.stat == 1 || solution.stat == -1
    CC_center = solution.full(1:dim);
    radius = solution.obj;
else
    solution
    error('Could not solve the LP, consult the information above.');
end
