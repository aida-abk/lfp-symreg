%% to-do

%%|
% 1. Plot every channel
% 2. High pass and low pass every channel at multiple frequencies
% |%%


Sampling_frequency = trialdata.info.Fs;
nChannels = size(trialdata.lfp, 2);
order = 4;
nyquist = Sampling_frequency / 2;


% 1.1 Plot every channel
figure;
for channel = 1:nChannels
    x = trialdata.lfp{1, channel};
    t = (1:length(x)) / Sampling_frequency;
    plot(t, x);
    hold on;
end
hold off;
xlabel('Time (s)');
ylabel('LFP(unknown)');

% 1.2. Grid of all 32 channels
figure;
nRows = ceil(sqrt(nChannels));
nCols = ceil(nChannels / nRows);
ax = gobjects(nChannels, 1);   % store axis handles

for channel = 1:nChannels
    x = trialdata.lfp{1, channel};
    t = (1:length(x)) / Sampling_frequency;
    ax(channel) = subplot(nRows, nCols, channel);
    plot(t, x);
    title(sprintf('Ch %d', channel));
    if channel > nChannels - nCols, xlabel('Time (s)'); end
    if mod(channel-1, nCols) == 0, ylabel('LFP(unknown)'); end
end
sgtitle('All 32 channels');

linkaxes(ax, 'y');   % all share one y-axis range (auto-fit to the widest)

% 2.1 High pass and low pass every channel at multiple frequencies

FS = tf.freqs(1:6);   % the cutoff frequencies I want to try
x  = trialdata.lfp{1, 1};
t  = (1:length(x)) / Sampling_frequency;

for f = 1:numel(FS)
    cutoff = FS(f);
    Wn = cutoff / nyquist;   % normalized cutoff

    [b_lp, a_lp] = butter(order, Wn, 'low'); %butterworth filter
    [b_hp, a_hp] = butter(order, Wn, 'high');

    lp = filtfilt(b_lp, a_lp, x);
    hp = filtfilt(b_hp, a_hp, x);

    figure;
    plot(t, x); hold on;
    plot(t, lp);
    plot(t, hp);
    hold off;
    legend('Raw', 'Low-pass', 'High-pass');
    title(sprintf('Channel 1 filtered at %.2f Hz', cutoff));
    xlabel('Time (s)'); ylabel('LFP (unknown)');
end

