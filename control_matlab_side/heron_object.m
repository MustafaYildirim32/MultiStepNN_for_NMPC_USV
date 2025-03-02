classdef heron_object
    properties
        Position            % [x; y; yaw]
        Velocity            % [u; v; r]

        % Handles for the dynamic robot plot
        PlotHandle          % Plot marker for Heron (current position)
        PathArrowHandles    % Array of handles for the arrows along the path trace
        YawHandle           % Quiver for heading

        % Path-related
        PathHandle          % Handle to the line that shows the path traveled
        Xhistory            % x-positions over time
        Yhistory            % y-positions over time

        % Reference points & circles
        RefPoints           % The reference points (pos_ref_points)
        PlotRefHandle       % Handle for plotting reference points
        PlotCircleHandles   % Array of handles for filled circles

        % Special markers
        StartPosHandle      % Start position marker
        FinalGoalHandle     % Final reference point marker

        % Optional angles, if needed
        Phi_Angle
        Anchor_Angle
    end

    methods
        %% Constructor
        function obj = heron_object(pos, vel, pos_ref_points)
            % Store position & velocity
            obj.Position = pos;
            obj.Velocity = vel;

            % Create a new figure and set up the axes
            figure('Name','Heron Position Control','NumberTitle','off');
            hold on; grid on;

            % -- Initialize the array for path arrows
            obj.PathArrowHandles = [];
            
            % -- 1) Mark the Start Position (now in cyan)
            obj.StartPosHandle = plot(pos(1), pos(2), 'co', ...
                'MarkerSize', 8, 'MarkerFaceColor','c', ...
                'DisplayName','Start');

            % -- 2) Plot the robot's current position (blue circle)
            obj.PlotHandle = plot(pos(1), pos(2), 'bo','MarkerSize', 10.5, ...
                'MarkerFaceColor','b', ...
                'DisplayName','Robot Position');

            % -- 3) Short yaw arrow (shows heading)
            obj.YawHandle = quiver(pos(1), pos(2), ...
                                   cos(pos(3)), sin(pos(3)), ...
                                   3, 'r', 'LineWidth',1.4, ...
                                   'MaxHeadSize',5, 'AutoScale','off', ...
                                   'DisplayName','Yaw Direction');

            % -- 4) Keep track of the path as a dashed elegant orange line
            obj.Xhistory = pos(1);
            obj.Yhistory = pos(2);
            obj.PathHandle = plot(obj.Xhistory, obj.Yhistory, '--', ...
                                  'Color',[1, 0.5, 0], 'LineWidth',1.2, ...
                                  'DisplayName','Robot Path');

            % -- 5) Plot the reference points & translucent forest green circles
            if exist('pos_ref_points','var') && ~isempty(pos_ref_points)
                obj.RefPoints = pos_ref_points;  % store them

                % Plot all reference points in black "x"
                obj.PlotRefHandle = plot(pos_ref_points(1,:), ...
                                         pos_ref_points(2,:), ...
                                         'kx', 'MarkerSize',4, ...
                                         'LineWidth',1.5, ...
                                         'HandleVisibility','off');  % No legend entry

                % Fill translucent forest green circles of radius=3
                r = 3;
                thetaCircle = linspace(0,2*pi,50);
                obj.PlotCircleHandles = gobjects(1, size(pos_ref_points,2));
                for i = 1 : size(pos_ref_points,2)
                    cx = pos_ref_points(1,i);
                    cy = pos_ref_points(2,i);
                    xCirc = r*cos(thetaCircle) + cx;
                    yCirc = r*sin(thetaCircle) + cy;
                    obj.PlotCircleHandles(i) = fill(xCirc, yCirc, [0, 0.4470, 0.7410], ...
                        'FaceAlpha', 0.2, ...      % translucent fill
                        'EdgeColor', [0, 0.4470, 0.7410], ...
                        'HandleVisibility','off');   % No legend entry
                end

                % -- 6) Mark the final reference point in deep purple
                finalRef = pos_ref_points(:, end); % last column
                obj.FinalGoalHandle = plot(finalRef(1), finalRef(2), 'o', ...
                    'MarkerSize', 8, 'MarkerFaceColor',[0.1333, 0.5451, 0.1333], ...
                    'MarkerEdgeColor',[0.1333, 0.5451, 0.1333], ...
                    'DisplayName','Final Goal');
            end

            % -- 7) Set axis limits and keep 1:1 scaling
            xlim([-50, 50]);
            ylim([-50, 50]);
            axis equal;

            % Add labels and title
            xlabel('x (meters)');
            ylabel('y (meters)');
            title('Heron Simulation');

            % Explicitly order the legend entries
            legend([obj.PlotHandle, obj.YawHandle, obj.PathHandle, obj.StartPosHandle, obj.FinalGoalHandle], ...
                {'Robot Position', 'Yaw Direction', 'Robot Path', 'Start', 'Final Goal'}, 'Location','best');

            hold off;
            drawnow;
        end

        %% Transformation matrix (optional)
        function T = transformationMatrix(obj)
            psi = obj.Position(3);
            T = [cos(psi), -sin(psi), 0; ...
                 sin(psi),  cos(psi), 0; ...
                 0,         0,        1];
        end

        %% Update the plot
        function obj = updatePlot(obj, refresh)
            % Append current position to history
            obj.Xhistory(end+1) = obj.Position(1);
            obj.Yhistory(end+1) = obj.Position(2);
            
            if refresh
                % 1) Update robot marker position
                set(obj.PlotHandle, 'XData', obj.Position(1), ...
                                    'YData', obj.Position(2));

                % 2) Update yaw arrow (heading indicator)
                set(obj.YawHandle, 'XData', obj.Position(1), ...
                                    'YData', obj.Position(2), ...
                                    'UData', cos(obj.Position(3)), ...
                                    'VData', sin(obj.Position(3)));

                % 3) Update path history (dashed elegant orange line)
                set(obj.PathHandle, 'XData', obj.Xhistory, ...
                                    'YData', obj.Yhistory);

                % 4) Every 50 steps, add a new arrow head to show the segment direction
                if mod(length(obj.Xhistory), 50) == 0 && length(obj.Xhistory) > 1
                    % Compute the direction of the arrow from the last two points
                    dx = obj.Xhistory(end) - obj.Xhistory(end-1);
                    dy = obj.Yhistory(end) - obj.Yhistory(end-1);
                    theta = atan2(dy, dx);
                    
                    % Define arrow head dimensions for a more wide open appearance:
                    arrowHeadLength = 3.2;    % Length from base to tip of the arrow head
                    arrowHeadWidth  = 1.6;      
                    offsetAngle     = pi/6;   % spread angle
                    
                    % Define the tip of the arrow head (current end position)
                    tipX = obj.Xhistory(end);
                    tipY = obj.Yhistory(end);
                    
                    % Compute the base point of the arrow head (moving back from tip)
                    baseX = tipX - arrowHeadLength * cos(theta);
                    baseY = tipY - arrowHeadLength * sin(theta);
                    
                    % Calculate left and right corners of the arrow head base
                    leftX  = baseX + arrowHeadWidth * cos(theta + offsetAngle);
                    leftY  = baseY + arrowHeadWidth * sin(theta + offsetAngle);
                    rightX = baseX + arrowHeadWidth * cos(theta - offsetAngle);
                    rightY = baseY + arrowHeadWidth * sin(theta - offsetAngle);
                    
                    % Draw the arrow head as a filled polygon in elegant orange
                    hold on;
                    fill([tipX, leftX, rightX], [tipY, leftY, rightY], [1, 0.5, 0], ...
                         'EdgeColor', [1, 0.5, 0], 'HandleVisibility', 'off');
                end

                % Keep axis limits and 1:1 aspect ratio
                xlim([-50, 50]);
                ylim([-50, 50]);
                axis equal;
                drawnow;
            end
        end

        %% Update the object state
        function obj = update(obj, newPosition, newVelocity, refresh)
            obj.Position = newPosition;
            obj.Velocity = newVelocity;
            obj = obj.updatePlot(refresh);
        end
    end
end
