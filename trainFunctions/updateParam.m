function [theta] = updateParam(theta,grad, trainParams)
    lr = trainParams.lr;
    if trainParams.GPU == true
        theta =    gpuArray(theta);
    end
    
    theta = theta - (lr*grad);
        if trainParams.GPU == true
            theta = gather(theta);
        end
    end