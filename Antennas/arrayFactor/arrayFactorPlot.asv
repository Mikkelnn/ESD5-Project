% Array Factor for Uniform Linear Array (ULA)

% User-defined parameters
N = 10;          % Number of elements in the array
d = 0.5;        % Spacing between elements (in wavelength units)
beta = 0;    %pi/2 Signal phase difference (in radians)

% Define angle range for plotting
theta = linspace(-pi, pi, 1000);  % Theta from -180 to 180 degrees

% Calculate Array Factor
AF = zeros(1, length(theta));     % Initialize Array Factor

for k = 1:length(theta)
    % Array Factor formula for ULA
    AF(k) = abs(sum(exp(1i * ( (0:N-1) * (beta - 2 * pi * d * cos(theta(k))) ))));
end

% Normalize the array factor
AF_normalized = AF / max(AF);

% Plot the Array Factor
figure;
polarplot(theta, AF_normalized, 'LineWidth', 2);
%title('Array Factor of Uniform Linear Array');
set(gca, 'ThetaZeroLocation', 'top');  % 0 degrees at top
set(gca, 'ThetaDir', 'clockwise');     % Angle direction clockwise
ax = gca;
ax.RLim = [0 1];

% Add grid and labels
grid on;
ax.ThetaTick = [-180:30:180];          % Set theta ticks
ax.RTickLabel = {'0', '', '0.5', '', '1'};  % Set radial ticks

% Additional Plot Settings
xlabel('Angle \theta (degrees)');
ylabel('Normalized Array Factor');

