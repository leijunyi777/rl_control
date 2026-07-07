figure

% ====== 创建视频对象 ======
v = VideoWriter('my_animation.mp4', 'MPEG-4');  % 输出文件名
v.FrameRate = 20;   % 视频帧率，可调
open(v);            % 打开视频开始写入
% ========================

for l = 1:s(2)
    clf

    % Step 3: Rotation matrix
    R = [cos(theta(l)) -sin(theta(l));
         sin(theta(l))  cos(theta(l))];
    corners_world = R * corners_local + p_center(l,:)';
    corners_world = [corners_world corners_world(:,1)];
    plot(corners_world(1,:), corners_world(2,:), 'r-', 'LineWidth', 2);
    hold on

    for i = 1:ne
        plot(ps(:, d*(i-1)+1),      ps(:, d*(i-1)+2), 'k');
        plot(ps(:, d*(i-1)+1)+la,   ps(:, d*(i-1)+2), 'k');
        hold on

        rectangle('Position', ...
           [ps(l, d*(i-1)+1)-w/2, ps(l, d*(i-1)+2)-L, w, L], ...
           'EdgeColor','b','LineWidth',2);

        axis equal
        % xlim([-4 4]);   % 如果你要固定 X 范围可以打开
    end

    drawnow

    % ====== 写入这一帧到视频 ======
    frame = getframe(gcf);
    writeVideo(v, frame);
    % =============================
end

% ====== 关闭视频文件 ======
close(v);
disp('视频已保存为 my_animation.mp4');
% ==========================

