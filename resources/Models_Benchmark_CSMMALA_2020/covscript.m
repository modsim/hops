wt_cov = eye(289, 289) * 0.025;

for row=1:size(wt_cov,1)
    for col=1:row % size(wt_cov,2)
          if col ~= row
             if rand(1, 1) < 0.5
                wt_cov(row, col) =  0.05 * (1 - 2*rand(1, 1)) * sqrt(wt_cov(row, row) * wt_cov(col, col));
                wt_cov(col, row) = wt_cov(row, col);
             end
          end
    end
end