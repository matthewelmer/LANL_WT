function TestFunc(G,Me,n)

%{
    Defines geometry for a minimum length nozzle based on a design exit
    mach number for a certain gas, given a finite number (n) of mach waves.
    Based on the information described in Anderson, Modern Compressible
    Flow 3rd Edition (Library of Congress CN: 2002067852).

Input parameters
    G is gamma, the ratio of specific heats (Cp/Cv)
    Me is the design exit mach number
    n is the finite number of expansion waves used in approximation
    
%}
    
%% Initialize datapoint matrices
Km = zeros(n,n);    % K- vlaues (Constant along right running characteristic lines)
Kp = zeros(n,n);    % K- vlaues (Constant along left running characteristic lines)
Theta = zeros(n,n); % Flow angles relative to the horizontal
Mu = zeros(n,n);    % Mach angles
M = zeros(n,n);     % Mach Numbers
x = zeros(n,n);     % x-coordinates
y = zeros(n,n);     % y-coordinates

%% Find NuMax (maximum angle of expansion corner)
[~, B, ~] = PMF(G,Me,0,0);
NuMax = B/2;

%% Define flow of first C+ lines in expansion section
[xList, yList] = Expansion(.5, deg2rad(NuMax), n);
dT = NuMax/n;

Theta(:,1) = (dT:dT:NuMax);

Nu = Theta;
Km = Theta + Nu;
Kp = Theta - Nu;
[M(:,1) Nu(:,1) Mu(:,1)] = PMF(G,0,Nu(:,1),0);
plot(xList, yList);
hold on
% xList
% yList
for i = 1:length(yList)
    xNew = xList(i) - yList(i)/tand(Theta(i,1)-Mu(i,1));
    xTemp = [xList(i) xNew];
    yTemp = [yList(i) 0];
    plot(xTemp, yTemp);
    hold on
%     xNew
axis equal
end