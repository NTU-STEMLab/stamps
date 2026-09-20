# -*- coding: utf-8 -*-
from six.moves import range
import numpy as np



def diffarray(x, r=None, include=True, aniso=False, gui_args=()):
    '''
    Create the unique pair index across the coordinates of x

    Syntax: index_head,index_tail,diff = diffarray(x,r=None,include=True)

    Input:    

    x           n by nd     2D array    row for data and col for dimension
    r           scalar                  the range to be considered
    include     bool                    True denotes diff include self
    aniso       bool                    True consider anisotropy 
                                        (x.shape[1] must be 2)
    gui_args                tuple       a tuple contains gui elements
    Output:

    index_head  n           1D array    the first index of the pairs
    index_tail  n           1D array    the second index of the pairs
    diff        n by nd     2D array    the differerces between the pairs at
                                          each dimension
    angdiff     n           1D array    the angles in radius in which the
                                        horizontal x-direction are considered to
                                        be zero degrees
    '''

    row_count, col_count = x.shape
    d = 0 if include == True else 1
    if col_count != 2 and aniso == True: # error 
        raise ValueError("aniso is True but x's dimension is not 2")
    k = 0
    nn0 = int((row_count - d)*(row_count - d + 1)/2)
    hasMemoryError = False
    try:
        #index_head = np.zeros(nn0)
        index_head = np.zeros(int(nn0))
        #index_tail = np.zeros(nn0)
        index_tail = np.zeros(int(nn0))
        #diff = np.zeros((nn0, col_count))
        diff = np.zeros((int(nn0), int(col_count)))
        #angdiff = np.zeros(nn0) if aniso else None
        angdiff = np.zeros(int(nn0)) if aniso else None
    except MemoryError as memerr:
        hasMemoryError = True

    if r is None:
        if hasMemoryError:
            raise memerr # we have no solution here...
        else:
            for i in range(row_count - d):
                index_head[k:k+row_count-i-d] = i
                index_tail[k:k+row_count-i-d] = np.arange(row_count)[i+d:]
                diff[k:k+row_count-i-d]=x[i]-x[np.arange(row_count)[i+d:]]
                if aniso is True:
                    # Use arctan2 to avoid division-by-zero when dx == 0
                    # (vertical pairs).  This is consistent with the r-limited
                    # branch below which already uses arctan2.
                    ang_slice = np.arctan2(
                        diff[k:k+row_count-i-d][:, 1],
                        diff[k:k+row_count-i-d][:, 0],
                    )
                    # Normalize to (-pi/2, pi/2] so that the same spatial
                    # direction is assigned the same bin regardless of which
                    # endpoint is considered "head" vs "tail" (D2 fix).
                    ang_slice[ang_slice > np.pi / 2] -= np.pi
                    ang_slice[ang_slice <= -np.pi / 2] += np.pi
                    angdiff[k:k+row_count-i-d] = ang_slice
                k = k + row_count - i - d
            index_head = index_head.astype(int)
            index_tail = index_tail.astype(int)
    else: # consider r
        if hasMemoryError: # try split x
            print ('Try to split x...')
            index_head = np.array([], dtype=np.int64)
            index_tail = np.array([], dtype=np.int64)
            diff = np.array([]).reshape((0, col_count))
            angdiff = np.array([]) if aniso else None
            step_len = 100
            step_count = int(np.ceil(float(row_count) / step_len))
            current_count = 0
            if gui_args:
                title = DataObj.getProgressText()
                gui_args[0].setProgressRange(0, step_count)
                sub_title = title + "\n- Diffarray..."
                gui_args[0].setCurrentProgress(0, sub_title)
            for step in range(0, row_count, step_len):
                piece_x = x[step:step+step_len, :]
                for step2 in range(step, row_count, step_len):
                    piece_x2 = x[step2:step2+step_len, :]
                    if step == step2: #need upper-triangle
                        idx_head_step, idx_tail_step, diff_step, angdiff_step =\
                            diffarray(piece_x, r, include, aniso)
                    else: #need square
                        idx_head_step, idx_tail_step, diff_step, angdiff_step =\
                            cdist(piece_x, piece_x2, r, aniso)
                    idx_head_step += step
                    idx_tail_step += step2
                    index_head = np.append(index_head, idx_head_step)
                    index_tail = np.append(index_tail, idx_tail_step)
                    diff = np.append(diff, diff_step, axis=0)
                    angdiff = np.append(angdiff, angdiff_step) if aniso else None
                    current_count+=1
                    if gui_args and not gui_args[0].wasProgressCanceled():
                        gui_args[0].setCurrentProgress(
                            current_step_count ,
                            sub_title + '({c}/{n})'.format(
                                c=current_count,
                                n=step_count
                                )
                            ) #rest
                        gui_args[0].drawGUI()
                    print (str(current_count)+ '/' +str(step_count*(step_count+1)/2))
                    print ('size:', index_head.shape)
        else:
            nn = nn0
            for i in range(row_count):
                index_head_=np.zeros(row_count-i-d)
                index_tail_=np.zeros(row_count-i-d)
                diff_=np.zeros((row_count-i-d,col_count))
                index_head_[:]=i
                index_tail_[:]=np.arange(row_count)[i+d:]
                diff_=x[i]-x[np.arange(row_count)[i+d:]]
                if aniso is True:
                    angdiff_ = np.arctan2(diff_[:,1], diff_[:,0])
                    # Normalize to (-pi/2, pi/2] (D2 fix).
                    angdiff_[angdiff_ > np.pi / 2] -= np.pi
                    angdiff_[angdiff_ <= -np.pi / 2] += np.pi
                else:
                    angdiff_ = None
                idxx = np.where(
                    np.sqrt((diff_**2).sum(axis=1, keepdims=True))<=r
                    )[0]
                ni = idxx.size
                index_head[k:k+ni] = index_head_[idxx]
                index_tail[k:k+ni] = index_tail_[idxx]
                diff[k:k+ni,:] = diff_[idxx]
                if aniso is True:
                    angdiff[k:k+ni] = angdiff_[idxx]
                k = k + ni
                if nn-k <= row_count-i:
                    index_head = np.append(index_head, np.empty(nn0))
                    index_tail = np.append(index_tail, np.empty(nn0))        
                    diff=np.vstack((diff,np.empty((nn0, col_count))))
                    if aniso is True:
                        angdiff = np.append(angdiff, np.empty(nn0))
                    nn = nn + nn0

            index_head=index_head[:k].astype(int)
            index_tail=index_tail[:k].astype(int)
            diff=diff[:k]
            if aniso:
                angdiff=angdiff[:k]
    
    return index_head, index_tail, diff, angdiff

def diffarray_split(x, r=None,
    include=True, aniso=False,
    split_length=3000, container_size_step=1000000, show_info=False,
    gui_args=()):
    '''solve memory problem'''

    def _enlarge_container(
        index_head, index_tail, diff, angdiff,
        maximum_container_size, container_size_step):
        if show_info:
            print ('enlarge containter')
        index_head = np.append(
            index_head, np.empty(
                container_size_step, dtype=np.int64
                )
            )
        index_tail = np.append(
            index_tail, np.empty(
                container_size_step, dtype=np.int64
                )
            )        
        diff = np.vstack(
            (diff, np.empty((container_size_step, col_count)))
            )
        if aniso is True:
            angdiff = np.append(
                angdiff, np.empty(
                    container_size_step, dtype=np.int64
                    )
                )
        maximum_container_size += container_size_step
        return (index_head, index_tail, diff, angdiff,
            maximum_container_size)

    row_count, col_count = x.shape
    d = 0 if include == True else 1
    if col_count != 2 and aniso == True: # error 
        raise ValueError("aniso is True but x's dimension is not 2")

    index_head = np.empty(container_size_step, dtype=np.int64)
    index_tail = np.empty(container_size_step, dtype=np.int64)
    diff = np.empty((container_size_step, col_count))
    angdiff = np.empty(container_size_step) if aniso else None
    step_len = split_length
    step_count = int(np.ceil(float(row_count) / step_len))
    total_step = (step_count+1) * step_count // 2
    current_count = 0
    current_data_size = 0
    maximum_container_size = container_size_step

    if gui_args:
        title = gui_args[0].getProgressText()
        gui_args[0].setProgressRange(0, total_step)
        sub_title = '\n'.join((title, gui_args[1]))
        gui_args[0].setCurrentProgress(0, sub_title)

    for step in range(0, row_count, step_len):
        piece_x = x[step:step+step_len, :]
        for step2 in range(step, row_count, step_len):
            piece_x2 = x[step2:step2+step_len, :]
            if step == step2: #need upper-triangle
                idx_head_step, idx_tail_step, diff_step, angdiff_step =\
                    diffarray(piece_x, r, include, aniso)
            else: #need square
                idx_head_step, idx_tail_step, diff_step, angdiff_step =\
                    cdist(piece_x, piece_x2, r, aniso)
            idx_head_step += step
            idx_tail_step += step2

            data_size_step = idx_head_step.size
            current_data_size += data_size_step
            while current_data_size > maximum_container_size:
                (index_head, index_tail, diff, angdiff,
                maximum_container_size) =\
                    _enlarge_container(
                        index_head, index_tail, diff, angdiff,
                        maximum_container_size, container_size_step)
            slice_idx = slice(
                current_data_size-data_size_step, current_data_size)
            index_head[slice_idx] = idx_head_step
            index_tail[slice_idx] = idx_tail_step
            diff[slice_idx, :] = diff_step
            if aniso:
                angdiff[slice_idx] = angdiff_step

            # index_head = np.append(index_head, idx_head_step)
            # index_tail = np.append(index_tail, idx_tail_step)
            # diff = np.append(diff, diff_step, axis=0)
            # angdiff = np.append(angdiff, angdiff_step) if aniso else None
            current_count+=1
            if show_info:
                print (str(current_count)+ '/' +str(step_count*(step_count+1)/2))
            if gui_args:
                if not gui_args[0].wasProgressCanceled():
                    gui_args[0].setCurrentProgress(
                        current_count ,
                        sub_title + '({c}/{n})'.format(
                            c=current_count,
                            n=total_step
                            )
                        ) #rest
                    gui_args[0].drawGUI()
                else:
                    return False
    if gui_args:
        gui_args[0].setCurrentProgress(text=title)
    slice_final = slice(0, current_data_size)
    index_head = index_head[slice_final]
    index_tail = index_tail[slice_final]
    diff = diff[slice_final, :]
    if aniso:
        angdiff = angdiff[slice_final]

    return index_head, index_tail, diff, angdiff

def multiarray(x,include=True):
    '''
    like diffarray but operator is multilize
    '''
    lenx=len(x)
    diff=[]

    index_head=[]
    index_tail=[]
        
    if include == True:
        for i in range(lenx):
            index_head+=[i]*(lenx-i)
            index_tail+=range(lenx)[i:]
        for i in range(lenx):  
            for j in range(i,lenx):
                #index_head.append(i)
                #index_tail.append(j)
                diff.append(x[i]*x[j])
    elif include == False:
        for i in range(lenx-1):
            index_head+=[i]*(lenx-i-1)
            index_tail+=range(lenx)[i+1:]
        for i in range(lenx): 
            for j in range(i+1,lenx):
                #index_head.append(i)
                #index_tail.append(j)
                diff.append(x[i]*x[j])
    
    index_head,index_tail,diff=list(map(np.array,(index_head,index_tail,diff)))
    return index_head,index_tail,diff

def cdist(x, y, r, aniso):
    x_row, x_col = x.shape
    y_row, y_col = y.shape
    if x_col != y_col:
        raise ValueError(
            'two array must have the same column length')
    if x_col != 2 and aniso == True:
        raise ValueError(
            'anisotropy is only suit for column length equal to 2')

    index_head = np.empty( x_row * y_row, dtype=np.int64)
    index_tail = np.empty( x_row * y_row, dtype=np.int64)
    for i in range(x_row):
        index_head[i*y_row:(i+1)*y_row] = i
        index_tail[i*y_row:(i+1)*y_row] = np.arange(y_row)


    diff = np.empty((x_row*y_row, x_col))
    for i in range(x_col):
        diff[:,i] = (x[:,i:i+1] - y[:,i]).ravel()
    if aniso == True:
        angdiff = np.arctan2(diff[:,1], diff[:,0]).ravel()
    else:
        angdiff = np.empty(x_row*y_row)

    if r != -1.0:
        dis_arr = np.zeros(x_row*y_row)
        for i in range(x_col):
            dis_arr = dis_arr+diff[:,i]**2

        idxx = np.where(np.sqrt(dis_arr) <= r)
        index_head = index_head[idxx]
        index_tail = index_tail[idxx]
        diff = diff[idxx]
        if aniso == True:
            angdiff = angdiff[idxx]

    return index_head.ravel(), index_tail.ravel(), diff, angdiff.ravel()

if __name__ == "__main__":
    import time

    x = np.array(np.random.random((10,2)))
    r = 0.5
    include = True
    aniso = True
    stime=time.time()
    h1, t1, d1, a1 = diffarray(x, r, include, aniso)
    print ('No split time cost:', time.time() - stime)
    stime=time.time()
    h1s, t1s, d1s, a1s = diffarray_split(x, r, include, aniso,show_info=True)
    print ('Split time cost:', time.time() - stime)
