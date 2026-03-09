clear
clc
tic;
% format rat
format short
%%系统参数
%论文中例一
A = [1,0.1,0,0;0,0.9818,0.2673,0;0,0,1,0.1;0,-0.0455,3.1182,1];
B = [0;0.1818;0;0.4546];
epsilon = 1e-3;
% 创建MPT模型
sys = LTISystem('A', A, 'B', B);
xmin = [-1;-1.5;-0.35;-1];
xmax = [1;1.5;0.35;1];
X = Polyhedron('lb',xmin,'ub', xmax);%初始集合
U = Polyhedron('lb',-1,'ub',1);

% 计算控制不变集
num_of_iteration = 8;
Omega = X;
C_infinity = X;
for k = 1:num_of_iteration
    Pre_Omega = sys.reachableSet('X',C_infinity,'U',U,'N',1,'direction','backward');
    C_infinity_new = intersect(C_infinity, Pre_Omega);
    hausdorff_dist = C_infinity_new.distance(C_infinity).dist;
    fprintf('Iteration %d: Hausdorff Distance = %.6f\n', k, hausdorff_dist);
    % if hausdorff_dist < epsilon
    %     disp('控制不变集收敛');
    %     break;
    % end
    C_infinity = C_infinity_new;
end


%四维时候画投影
% 获取控制不变集的顶点
CIS_A = C_infinity.A;
CIS_b = C_infinity.b;

vertices = C_infinity.V;

% 需要投影的维度组合
projection_pairs = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];

% 创建图形窗口
figure;

for i = 1:6
    % 选择投影的两个维度
    x_idx = projection_pairs(i, 1);
    y_idx = projection_pairs(i, 2);
    
    % 提取对应维度的点
    x_data = vertices(:, x_idx);
    y_data = vertices(:, y_idx);
    
    % 计算二维投影的凸包（获取边界）
    K = convhull(x_data, y_data);
    
    % 绘制凸包的多面体投影
    subplot(2, 3, i);
    fill(x_data(K), y_data(K), 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'k', 'LineWidth', 1.5);  % 只绘制凸包的面
    xlabel(['x', num2str(x_idx)]);
    ylabel(['x', num2str(y_idx)]);
    title(['Projection on x', num2str(x_idx), ' - x', num2str(y_idx)]);
    xlim([-0.1+xmin(projection_pairs(i,1)), 0.1+xmax(projection_pairs(i,1))]);  % 设置 x 轴的范围
    ylim([-0.1+xmin(projection_pairs(i,2)), 0.1+xmax(projection_pairs(i,2))]);
    grid on;
    % axis equal;
end
sgtitle('The projection of the CIS for the four-dimensional system');
part1time = toc;
fprintf('运行时间为:%.4f 秒\n', part1time);
