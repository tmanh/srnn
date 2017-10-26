function mat2ske()
    src = '/home/anhtruong/Downloads/Dataset/Interaction/Human-Human/M2I/Human_Human';
    dst = 'M2I';
    dataset2folder(src, dst);
end

function dataset2folder(src, dst)
    list = dir(src);
    list=list(~ismember({list.name},{'.','..'}));
    [len, ~] = size(list);
    
    if ~exist(dst,'dir')
        mkdir(dst);
    end
    
    for i = 1:len
        if ~isdir([list(i).folder '/' list(i).name])
            continue
        end
        
        path = [list(i).folder '/' list(i).name];
        activity2folder(path, [dst '/' list(i).name]);
    end
end

function activity2folder(src, dst)
    list = dir(src);
    list=list(~ismember({list.name},{'.','..'}));
    [len, ~] = size(list);
    
    if ~exist(dst,'dir')
        mkdir(dst);
    end
    
    for i = 1:len
        if ~isdir([list(i).folder '/' list(i).name])
            continue
        end
        
        path = [list(i).folder '/' list(i).name];
        view2folder(path, [dst '/' list(i).name]);
    end
end

function view2folder(src, dst)
    feature = '/Txt_Skel';
    list = dir([src feature]);
    list=list(~ismember({list.name},{'.','..'}));
    [len, ~] = size(list);
    
    if ~exist(dst,'dir')
        mkdir(dst);
    end
    
    for i = 1:len
        if ~isdir([list(i).folder '/' list(i).name])
            continue
        end
        
        path = [list(i).folder '/' list(i).name];
        ske2file(path, [dst '/' list(i).name '.txt']);
    end
end

function ske2file(src, dst)
    array = concatSke(src);
    
    if isempty(array)
        return
    end
    
    dlmwrite(dst, array, '\t');
end

function array = concatSke(folder)
    path = [folder '/*.mat'];
    
    list = dir(path);
    [len, ~] = size(list);
    
    array = [];
    for i = 1:len
        if i == 1
            array = getArray([list(i).folder '/' list(i).name]);
        else
            tmp = getArray([list(i).folder '/' list(i).name]);
            array = cat(1, array, tmp);
        end
    end
end

function array = getArray(path)
    Data = load(path);
    DataField = fieldnames(Data);
    col = Data.(DataField{1});
    [sz1, sz2] = size(col);
    array = reshape(col, [sz2, sz1]);
end