track2_path = 'data/11triv/components/x/track2_permanence.msgpack';

bytes = fileread_bytes(track2_path);
s = parsemsgpack(bytes);
t = cellnumvec(s('frameTimes_s'));
X = cellnummat(s('xPositions'));

figure;
plot(t, X, 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Value');
title(strrep(track2_path, '_', '\_'));
grid on;

function bytes = fileread_bytes(path)
fid = fopen(path, 'rb');
assert(fid >= 0, 'Could not open file: %s', path);
c = onCleanup(@() fclose(fid));
bytes = fread(fid, inf, '*uint8');
end

function v = cellnumvec(c)
v = cellfun(@double, c(:));
end

function M = cellnummat(c)
nr = numel(c);
nc = numel(c{1});
M = nan(nr, nc);
for i = 1:nr
    M(i, :) = cellfun(@double, c{i});
end
end
