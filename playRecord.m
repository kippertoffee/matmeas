function [ y ] = playRecord( outID, inID, x, varargin)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

fs = 48e3;
bits = 16;
reclen = 0;
inChans = 0;

for i = 1 : length(varargin)
    if(strcmpi(varargin{i}, 'fs'))
        fs = varargin{i+1};
    elseif(strcmpi(varargin{i}, 'bits'))
        bits = varargin{i+1};
    elseif(strcmpi(varargin{i}, 'reclen'))
        reclen = varargin{i+1};
    elseif(strcmpi(varargin{i}, 'inchans'))
        inChans = varargin{i+1};
    end
end

if reclen == 0
    reclen = length(x) / fs;
end
if inChans == 0 % assumes less channels than samples
    inChans = min(size(x));
end


r = audiorecorder(fs, bits, inChans, inID);
p = audioplayer(x, fs, bits, outID);
record(r, reclen);
playblocking(p)

y = getaudiodata(r);

end

