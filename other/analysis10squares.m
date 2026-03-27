clear;close all
load data520b;
p=2;Tslide=10;
valid=find((t>50 & t<170) & ~isnan(xR(:,1)) & ~isnan(t));
t = t(valid);
t0=t;
ft0=0;ft02=0;
MagdB0{1}=0;
MagdB0{2}=0;
for ii=1:9
    d0 = sqrt(1*(xR(valid,ii)-xR(valid,ii+1)).^2+0*(yR(valid,ii)-yR(valid,ii+1)).^2); %longitudinal modes
    dk = sqrt(0*(xR(valid,ii)-xR(valid,ii+1)).^2+1*(yR(valid,ii)-yR(valid,ii+1)).^2); %transverse modes
%% signal and Fourier transform
for jj=1:2
    if jj==1, t=t0;d=d0;else t=t0;d=dk;end
% If tracking dropped frames, interpolate onto a uniform time grid for FFT
dt = median(diff(t));                 % seconds
tU = (t(1):dt:t(end)).';              % uniform time vector
dU = interp1(t, d, tU, "linear");     % interpolate distance onto uniform grid

% 2) Detrend + window (reduces DC + spectral leakage)
dU = dU - mean(dU, "omitnan");        % remove DC
dU = detrend(dU);                     % remove linear trend
w  = hann(length(dU));                % window
x  = dU .* w;
tUt{jj}=tU;
dUt{jj}=dU;

% 3) FFT
Fs = 1/dt;                            % sampling frequency (Hz)
N  = length(x);
X  = fft(x);

% One-sided spectrum
P2 = abs(X / N);
P1 = P2(1:floor(N/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
Pt{jj,ii}=P1;
f = Fs*(0:floor(N/2))/N;
ft{jj,ii}=f;

% 4) Plot time signal + spectrum
figure(10*ii+1);
plot(tUt{jj}, dUt{jj}, "LineWidth", 1.5);hold on
grid on;
xlabel("Time (s)");
ylabel("Distance (pixels, detrended)");
title("Distance signal (prepared for FFT)");

figure(10*ii+2);
semilogy(ft{jj,ii}.^p, Pt{jj,ii},"LineWidth", 1);hold on
grid on;
xlabel("Frequency (Hz)");
ylabel("Amplitude (a.u.)");
title("One-sided amplitude spectrum of distance");
if jj==1, ft0=ft0+P1;f0=f;end
if jj==2, ft02=ft02+P1;f02=f;end

%% 2) Sliding window parameters
winLengthSec = Tslide;                    % Tslide-second window
winLength = round(winLengthSec * Fs); % samples
overlap = round(0.9 * winLength);     % 90% overlap for smooth evolution
hop = winLength - overlap;

n = length(dU);
nWindows = floor((n - winLength)/hop) + 1;

freqTrack = nan(nWindows,1);
timeTrack = nan(nWindows,1);

w = hann(winLength);

%% 3) Sliding FFT
for k = 1:nWindows
    idxStart = (k-1)*hop + 1;
    idxEnd   = idxStart + winLength - 1;

    segment = dU(idxStart:idxEnd) .* w;

    N = length(segment);
    X = fft(segment);

    P2 = abs(X/N);
    P1 = P2(1:floor(N/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);

    f = Fs*(0:floor(N/2))/N;

    % Ignore DC (index 1)
    [~, idxMax] = max(P1(2:end));
    freqTrack(k) = f(idxMax+1);

    timeTrack(k) = mean(tU(idxStart:idxEnd));
end

%% 2) Sliding window STFT (Tslide s window)
winSec   = Tslide;
winSamp  = max(8, round(winSec*Fs));  % samples in window (>=8 safety)
overlap  = round(0.9*winSamp);        % 90% overlap for smooth time axis
nfft     = 2^nextpow2(winSamp);       % FFT length (zero-padding)

w = hann(winSamp);

% spectrogram returns:
%   S: complex STFT (freq x time)
%   F: frequency vector (Hz)
%   T: time vector (s) relative to start of dU
[S,F,T] = spectrogram(dU, w, overlap, nfft, Fs);

% Convert to magnitude (or power)
Mag = abs((S));                         % amplitude
%Pow = (abs(S)).^2;                   % power (optional)

% Convert to dB for visualization (recommended)
MagdB{jj} = 20*log10(Mag + eps);
MagdB0{jj}=MagdB{jj}+MagdB0{jj};

% Shift time axis to actual time
T = T + tU(1);
Tt{jj}=T;
Ft{jj}=F;

%% 3) Surface plot (time-frequency magnitude)
%t = tiledlayout(1, 2, "TileSpacing", "tight");
figure(ii*10+3);
subplot(1,2,jj)
MagdBav{jj,ii}=mean(MagdB{jj},2);
Ftav{jj,ii}=Ft{jj};

%nexttile(t)
surf(Tt{jj}, Ft{jj}.^p, MagdB{jj}, "EdgeColor", "none");   % smooth surface
view(2);                                  % look from above (heatmap-like)
colormap jet
axis tight;
xlabel("Time (s)");
title("Sliding Fourier Transform (20 s window) - Magnitude (dB)");
colorbar
if jj==2, colorbar;
%hold on
%surf(Tt{jj}, 450-Ft{jj}.^p, MagdB{jj}, "EdgeColor", "none");   % smooth surface
view(2);                                  % look from above (heatmap-like)
colormap jet
axis tight;
else ylabel("Frequency (Hz)"); 
end
axis([0 max(T) 0 700])
caxis([0 50])

end

end

%%
figure;
semilogy(f0.^p, ft0, "LineWidth", 1.5);
hold on
semilogy(f02.^p, ft02, "LineWidth", 1.5);
grid on;
xlabel("Frequency (Hz)");
ylabel("Amplitude (a.u.)");
title("One-sided amplitude spectrum of distance");

%% Total Surface plot (time-frequency magnitude)
figure
for jj=1:2
subplot(1,2,jj)
surf(Tt{jj}, Ft{jj}.^p, MagdB0{jj}, "EdgeColor", "none");   % smooth surface
view(2);                                  % look from above (heatmap-like)
colormap jet
axis tight;
xlabel("Time (s)");
title("Sliding Fourier Transform (20 s window) - Magnitude (dB)");colorbar
if jj==2, colorbar;
%hold on
%surf(Tt{jj}, 450-Ft{jj}.^p, MagdB0{jj}, "EdgeColor", "none");   % smooth surface
view(2);                                  % look from above (heatmap-like)
colormap jet
axis tight;
else ylabel("Frequency (Hz)"); 
end
axis([0 max(T) 0 700])
caxis([0 400])
end
%%

col=jet(9);
freqs=[11.5 40.5 81.5 146 280 425 472 508 555];
    figure;
for ii=1:9
    plot(Ftav{1,ii}.^2,ii*10+smooth((MagdBav{1,ii}),1),'color',col(ii,:));hold on;
    for jj=1:9;
        int=abs(Ftav{1,ii}.^2-freqs(jj))<2;
        peaks(ii,jj)=mean(MagdBav{1,ii}(int));
    %[pks,locs] = findpeaks(y)
    end
end
xline(freqs)
figure
for jj=1:9
    w=1;
    if jj==5, w=4;end
plot(1:9,peaks(:,jj),'color',col(jj,:),'LineWidth',w)
hold on
end
xlabel('position')
ylabel('amplitude')
legend(num2str([1:9]'))

