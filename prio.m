classdef prio < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        fs = 48e3;
        availableDevices;
        outDevice = -1;
        outMaxChan = -1;
        outChans = [];
        isOutSet = false;
        
        inDevice = -1;
        inMaxChan = -1;
        inChans = [];
        isInSet = false;
        
        outBuffer = [];
        inBuffer = [];
        
        prefilter = [];
        
        isOutBufferCalibrated = 0;
        
        framesPerBuffer = 0;
        isInit = false;
        pages = [];
        
        cal = {};
        prefilt = {};
    end
    
  methods (Access = public)

    function IO = prio()
        IO.initCal();
    end

    function delete(IO)
        IO.reset();
    end

    function reset(IO)
        if playrec('isInitialised')
            playrec('reset');
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function a = list(IO)
        a = playrec('getDevices');
        IO.availableDevices = a;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % selecIo
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function selectIo(IO, inid, inchans, outid, outchans)
        IO.reset();
        IO.initCal();
        IO.selectIn(inid, inchans);
        IO.selectOut(outid, outchans);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % selectIn
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function selectIn(IO, inId, inChans)
        if isempty(IO.availableDevices)
            IO.list();
        end

        for d = IO.availableDevices
            if d.deviceID == inId
                if max(IO.inChans) > d.inputChans
                    error('Max input channel greater than number available for this device')
                end
                IO.cal.in.device = d.name;
            end
        end
        IO.inDevice = inId;
        IO.inChans = inChans;
        IO.inMaxChan = max(inChans);
        IO.isInSet = true;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % selectOut
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function selectOut(IO, outId, outChans)
        if isempty(IO.availableDevices)
            IO.list();
        end
        for d = IO.availableDevices
            if d.deviceID == outId
                if max(outChans) > d.inputChans
                    error('Max output channel greater than number available for this device')
                end
                IO.cal.out.device = d.name;
            end
        end
        IO.outDevice = outId;
        IO.outChans = outChans;
        IO.outMaxChan = max(outChans);
        IO.isOutSet = true;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % init
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function init(IO)
        if playrec('isInitialised')
            disp('#    Playrec already initialised; resetting.');
            playrec('reset');
        end
        if ~IO.isInSet && ~IO.isOutSet
            error('No device setup for in or output')
        end

        if ~isempty(dir('prioCalibration.mat'))
            load('prioCalibration.mat')
            if IO.channelsInArray(c.in.chans, IO.inChans)
                IO.cal = c;
                fprintf('#    Calibration file with matching IO found in working directory and loaded\n');
            end
        end

        playrec('init', IO.fs, IO.outDevice, IO.inDevice, IO.outMaxChan, IO.inMaxChan, IO.framesPerBuffer);
        IO.isInit = true;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % addOutAudio
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function addOutAudio(IO, x, outChans)
        if ~exist('outChans', 'var') || isempty(outChans)
            outChans = IO.outChans;
        end
        [nSamp, nChans] = size(x);

        [inArray, idx] = IO.channelsInArray(outChans, IO.outChans);
        if ~inArray
            error('Some output channels out of range')
        end

        if nChans > numel(IO.outChans)
            x = x(:, 1:numel(IO.outChans));
            fprintf('#prio    Too many channels of audio given. Removing %i channels\n', nChans - numel(IO.outChans))
        elseif nChans < numel(IO.outChans)
            n = numel(IO.outChans) - nChans;
            x = [x, zeros(nSamp, n)];
            fprintf('#prio    Not enough channels of audio given. Padding with %i channels of silence\n', n)
        end

        IO.outBuffer = x; % overwrite to output buffer
        IO.isOutBufferCalibrated = 0;

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % addCalibrateOutAudio
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function addCalibrateOutAudio(IO, x, db, chans)
        if ~exist('chans', 'var'); chans = IO.outChans; end
        x = IO.calibrateOutAudio(x, db, chans);
        IO.addOutAudio(x, chans);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calibrateOutAudio
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function y = calibrateOutAudio(IO, x, db, chans)
        if ~exist('chans', 'var'); chans = IO.outChans; end

        [inArray, idx] = IO.channelsInArray(chans, IO.cal.out.chans);
        if ~inArray || ~all(IO.cal.out.isCalibrated(idx))
            error('!!prio    Not all output channels are calibrated')
        end
        if size(x, 2) ~= numel(chans)
            error('!!prio    Number of audio channel given not matching out channels'); 
        end
        
        N = size(x, 1);
        w = hann(N); w = 2 * w / sum(w);
        X = fft(x .* w);
        Px = X .* conj(X);
        
        % freq weighting if required
        if strcmpi(IO.cal.out.weighting, 'A')
          f = fftFreqs(N, IO.fs); 
          A = aWeightingDb(f);
          PA = 10.^(0.1*A);
          Px = Px .* PA; 
        end
        
        Pout = sum(Px(1:N/2+1), 1);
        Lpout = 10*log10(Pout) - IO.cal.out.db(idx);
        Lpdiff = db - Lpout;
        lindiff = 10^(0.05*Lpdiff);
        y = x .* repmat(lindiff, [size(x, 1), 1]);

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % play
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function play(IO, varargin)
        for i = 1 : length(varargin)
            if isnumeric(varargin{i}) && size(varargin{i}, 1) > 1
                IO.addOutAudio(varargin{i});
            end
        end

        playrecord(IO, 'play');

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % rec
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function rec(IO, duration, varargin)
        playrecord(IO, 'rec', 'duration', duration);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % playrec
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function playrec(IO, varargin)
        duration = 0;
        for i = 1 : length(varargin)
            if isscalar(varargin{i})
                duration = varargin{i};
            elseif isnumeric(varargin{i}) && size(varargin{i}, 1) > 1
                IO.addOutAudio(varargin{i});
            end
        end
        if duration == 0, duration = size(IO.outBuffer, 1); end
        playrecord(IO, 'playrec', 'duration', duration);

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % getAudio
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function y = getAudio(IO, chans)
        [inArray, idx] = ismember(chans, IO.inChans);
        if ~all(inArray)
            error('!!prio    You have requested input channels that were not selected with selectIO() or selectIn()')
        end
        y = IO.inBuffer(:, idx);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % getCalibratedAudio
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function y = getCalibratedAudio(IO, chans)
        if ~exist('chans', 'var'); chans = IO.inChans; end

        [inArray, idx] = IO.channelsInArray(chans, IO.inChans);

        if ~inArray
            error('!!prio    You have requested channels that are no represented in your current calibration file. Use getAudio() for uncalibrated audio.')
        end

        y = IO.getAudio(chans);

        for i = 1 : numel(chans)
            y(:, i) = y(:, i) * IO.cal.in.lin(idx(i));
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % checkReadyToRecord
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function checkReadyToRecord(IO, varargin)
        if ~IO.isInit
            error('!!prio    prio not initialised')
        end
        if nargin > 1 && isempty(varargin{1})
            if ~IO.channelsInArray(varargin{1}, IO.inChans);
                error('!!prio    Trying to record on input channels not initialised')
            end
        end
        if nargin > 2 && isempty(varargin{2})
            if ~IO.channelsInArray(varargin{2}, IO.outChans);
                error('!!prio    Trying to playback on output channels not initialised')
            end
        end
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calibrateIn
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function calibrateIn(IO, inChans, varargin)
        if ~exist('inChans', 'var'); inChans = IO.inChans; end  % pick first in channel if unspecified
        IO.checkReadyToRecord(inChans);
        [inArray, inChanIdx] = IO.channelsInArray(inChans, IO.inChans);
        if ~inArray; error('!!prio    Not all channels specified for calibration are in the inChans array'); end

        f = 1e3;
        L = 94;
        t = 5;
        if nargin > 2
            f = varargin{1};
        end
        if nargin > 3
            L = varargin{2};
        end

        IO.initCal();

        for k = 1 : numel(inChanIdx)
            q = input(sprintf('#prio    Put calibrator on channel %i and press enter\n', inChans(k)));

            IO.rec(t * IO.fs);
            N = size(IO.inBuffer, 1);
            w = hann(N); w = 2 * w / sum(w);                % hannig window normalised for half-spectrum
            Y = fft(w .* IO.inBuffer(:, inChanIdx(k)));
            P = Y .* conj(Y);
            fFFT = fftFreqs(size(P, 1), IO.fs);

            %%%% calibrate based on one frequency bin
            [~, fIdx] = min(abs(fFFT - f));
            Pcal = max(P(fIdx - 10 : fIdx + 10)); % just in case it is not EXACTLY 1kHz

            %%%% calibrate based on surrounding frequency bins within 10dB of the main bin
%                 [~, fIdx] = min(abs(fFFT - f));
%                 Pi = P(fIdx - 10 : fIdx + 10);
%                 pidx = Pi > max(Pi) / 10;
%                 Pcal = sum(Pi(pidx));

            %%%% calibrate based on broad-band power calculated in time
%             K = 1024;
%             w = hann(K); w = w / sum(w);
%             Pcal = mean(filter(w, 1, IO.inBuffer(:, inChans).^2));
            %%%%

            Lp = 10 * log10(Pcal); % 3 added to account for -ve frequencies in FFT

            IO.cal.in.db(k) = L - Lp;
            IO.cal.in.lin(k) = 10^(IO.cal.in.db(k) * 0.05);
            IO.cal.in.isCalibrated = true;
            IO.cal.in.datetime = datetime();
            IO.cal.in.Level = L;
            IO.cal.in.requency = f;



            h = figure;
            semilogx(fFFT, 10 * log10(P) + IO.cal.in.db(k))
            xlim([10 IO.fs/2]); ylim([0 100])
            xlabel('Freq (Hz)'); ylabel('dB SPL (re. 20\muPa)')
        end

        IO.cal.in.chans = inChans;

        c = IO.cal;
        q = input('#prio    Save the config? ', 's');

        if strcmpi(q, 'y')
            save('prioCalibration.mat', 'c') 
        end

        if ishandle(h), close(h); end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calibrateOut
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function calibrateOut(IO, inChans, outChans, varargin)

        if ~exist('inChans', 'var') || isempty(inChans); inChans = IO.inChans(1); end  % pick first in channel if unspecified
        if ~exist('outChans', 'var') || isempty(outChans); outChans = IO.outChans; end  % pick all output chans if unspecified
        if numel(inChans) == 1; inChans = ones(size(outChans)) * inChans; end
        if numel(inChans) ~= numel(outChans)
          error('!!prio    Number of input channels must be 1 or equal to the number of output channels')
        end

        weighting = 'Z';
        dbfs = -35;
        T = 10;

        for i = varargin
          if ischar(i{:})
            weighting = i{:};
          elseif isscalar(i{:}) && i{:} < 0 
            dbfs = i{:};
          elseif isscalar(i{:}) && i{:} <= 60
            T = i{:};
          end
        end

%             if ~exist('dbfs', 'var'); dbfs = -35; end
        if dbfs > -10; error('!!prio    Whoah there! dbfs is a bit high, are you sure?'); end
%             if ~exist('T', 'var'); T = 10; end

        IO.checkReadyToRecord(inChans, outChans);
        [~, inChanIdx] = IO.channelsInArray(inChans, IO.inChans);
        [~, outChanIdx] = IO.channelsInArray(inChans, IO.outChans);

        N = IO.fs * T;
        w = hann(N); w = 2 * w / sum(w);                  % window normalised so that sine with p-p amp = 1 ia at 0dB in +ve feqs 
        x =  (2 * rand(N, 1) - 1);                        % create output
        X = fft(x .* w);                                  % to freq
        Px = X .* conj(X);                                % power

        % a-weight power spectrum if required
        if strcmpi(weighting, 'A')
          f = fftFreqs(N, IO.fs); 
          A = aWeightingDb(f);
          PA = 10.^(0.1*A);
          Px = Px .* PA; 
        end

        Px = sum(Px(1:N/2+1, 1));                         % power in +ve freqs
        diffdb = -(10*log10(Px) - dbfs);                  % diff between pow(X) and dbfs
        x = x .* 10^(0.05 * diffdb);                      % scale x to match dbfs

        out = zeros(T * IO.fs, numel(IO.outChans));       % empty matrix for output

        for k = 1 : numel(outChans)
          out(:, outChanIdx(k)) = x;              
          IO.playrec(out);                                % play & record x
          y = IO.getCalibratedAudio(inChans(k));          
          Y = fft(y .* w);                    
          Py = Y .* conj(Y);

          % a-weight power spectrum if required
          if strcmpi(weighting, 'A')
            Py = Py .* PA; 
          end

          Py = sum(Py(1:N/2+1));                          % power in +ve freqs
          Lpy = 10*log10(Py);                             % in dB
          IO.cal.out.db(outChanIdx(k)) = dbfs - Lpy;      % diff is calibration           
          IO.cal.out.isCalibrated(outChanIdx(k)) = 1;
          fprintf('#prio    Output calibration for channel %i = %0.1fdB\n', outChans(k), IO.cal.out.db(outChanIdx(k)));
        end

        IO.cal.out.chans = IO.outChans;
        IO.cal.out.datetime = datetime();
        IO.cal.out.weighting = weighting;

        c = IO.cal;
        q = input('Save the config? ', 's');

        if strcmpi(q, 'y')
          save('prioCalibration.mat', 'c') 
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % equaliseResponse
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function equaliseResponse(IO, outChans, inChans, fl, fh, N)
      IO.checkReadyToRecord();
      if ~exist('N', 'var'), N = 256; end

      Nmeas = 4096;
      f = fftFreqs(Nmeas, IO.fs);
      [~, flIdx] = min(abs(f - fl));
      [~, fhIdx] = min(abs(f - fh));

      T = 20;
      G = -30;
      x = (rand(T*IO.fs, 1) * 2 - 1) * 10.^(0.05 * G);

      prefilt = {};

      for i = 1 : length(outChans)
        IO.playrec(x);
        y = IO.getAudio(1);
        Py = psav(y, Nmeas, hann(Nmeas), N/2);
        Pmean = mean(Py(flIdx:fhIdx));       % get mean power in freq range
        Py = Py ./ Pmean;                    % centre around 0dB
        Py(flIdx:fhIdx) = 1 ./ Py(flIdx:fhIdx);
        Py(1:flIdx-1) = 1;
        Py(fhIdx+1:end) = 1;
        Py = smoothOct(Py, 1/3, 'hann');            % 3rd oct smoothing
        Py = fftMirror(Py, 2, 1);
        h = fftshift(ifft(sqrt(Py)));
        h = h(Nmeas/2 - (N/2-1) : Nmeas/2 + N/2) .* hann(N);    % truncate i.r.
        prefilt.chan(i) = outChans(i);
        prefilt.h(:, i) = h;
      end

      IO.prefilt = prefilt;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % channelsInArray
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [flag, idx] = channelsInArray(IO, chans, ioarray)
      [inArray, idx] = ismember(chans, ioarray);
      flag = all(inArray);
    end
    
  end
    
  methods (Access = protected)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % playrecord
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function playrecord(IO, action, varargin)
      IO.checkReadyToRecord();
      duration = 0;
      outLevelCheck = 1;

      for i = 1 : length(varargin)
        if strcmpi(varargin{i}, 'duration')
          duration = varargin{i+1};
        elseif strcmpi(varargin{i}, 'outaudio')
          duration = IO.addOutAudio(varargin{i+1});
        elseif strcmp(varargin{i}, 'FML')
          outLevelCheck = 0;
        end
      end

      if duration == 0, duration = size(IO.outBuffer, 1); end

      % prefiltering - currently only works with one channel
      if ~isempty(IO.prefilt)
        tmpOutBuffer = filter(IO.prefilt.h, 1, IO.outBuffer);
        disp('#prio    Pre-filtering output');
      else
        tmpOutBuffer = IO.outBuffer;
      end

      if outLevelCheck
        P = mean(tmpOutBuffer.^2, 1);
        if any(10*log10(P) > -25)
          error('!!prio    Output too loud. Use "FML" argument to bypass check, if you really want to');
        end
      end


      if strcmpi(action, 'play')
        nPage = playrec('play', tmpOutBuffer, IO.outChans);
        IO.pages(end+1,1) = nPage;
      elseif strcmpi(action, 'rec')
        nPage = playrec('rec', duration, IO.inChans);
        IO.pages(end+1,1) = nPage;
      elseif strcmpi(action, 'playrec')    
        nPage = playrec('playrec', tmpOutBuffer, IO.outChans, duration, IO.inChans);
        IO.pages(end+1,1) = nPage;
      end
      disp('#prio    Starting play/record....')
      playrec('block', IO.pages(1));
      disp('#prio    Ending play/record....')
      IO.outBuffer = [];            
      IO.inBuffer = double(playrec('getRec', IO.pages(1)));         % cast to double
      playrec('delPage', IO.pages(1));

      IO.pages = IO.pages(2:end, 1);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % initCal
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function initCal(IO)
      IO.cal.in.db = [];
      IO.cal.in.lin = [];
      IO.cal.in.isCalibrated = false;
      IO.cal.in.datetime = 0;
      IO.cal.in.level = 0;
      IO.cal.in.frequency = 0;
      IO.cal.in.chans = [];
      IO.cal.in.device = 0;

      IO.cal.out.db = [];
      IO.cal.out.lin = [];
      IO.cal.out.isCalibrated = false;
      IO.cal.out.datetime = 0;
      IO.cal.out.level = 0;
      IO.cal.out.frequency = 0;
      IO.cal.out.chans = [];
      IO.cal.out.device = 0;
      IO.cal.out.weighting = 'Z';
    end
  end
    
end

