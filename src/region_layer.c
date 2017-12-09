#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l = {0};
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + coords + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.coords = coords;
    //l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n*2, sizeof(float));
    l.biases_h = calloc(n*2, sizeof(int16_t));
    //l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30*(l.coords + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.delta_h = calloc(batch*l.outputs, sizeof(int16_t));
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.output_h = calloc(batch*l.outputs, sizeof(int16_t));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.forward_h = forward_region_layer_h;
    l.backward = backward_region_layer;


    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}


void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class >= 0){
            pred *= output[index + stride*class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {
        if (delta[index] && tag){
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);
            return;
        }
        for(n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            if(n == class) *avg_cat += output[index + stride*n];
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_region_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array(l.output + index,   l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax){
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }

    //memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
   
}
void forward_region_layer_h(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output_h, net.input_h, l.outputs*l.batch*sizeof(int16_t));
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_h(l.output_h + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array_h(l.output_h + index,   l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array_h(l.output_h + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax){
      cvt_single_prec (l.output_h, l.output, l.outputs*l.batch);
      cvt_single_prec (net.input_h, net.input, net.inputs * net.batch);
      int index = entry_index(l, 0, 0, l.coords + !l.background);
      softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
      //cvt_half_prec (net.input, net.input_h, net.inputs * net.batch);
    }

    //for (int i = 0; i < l.outputs*l.batch; i++)
    //printf("%d %.3f\n", i, l.output[i]);
    //printf("%x\n", l.output);
    /* for (int i = 0; i < l.outputs*l.batch; i++) */
    /*   printf("%d %.3f\n", i, l.output[i]); */
    

    //memset(l.delta_h, 0, l.outputs * l.batch * sizeof(int16_t));
    if(!net.train) return;

}

void backward_region_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}

void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}

void get_region_boxes(layer l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, float **masks, int only_objectness, int *map, float tree_thresh, int relative)
{
    int i,j,n,z;
    float *predictions = l.output;
    //printf("%x\n", predictions);
    //    printf("%.3f %.3f %.3f\n", predictions[0], predictions[1], predictions[200]);
    //for (i = 0; i < l.outputs*l.batch; i++)
    //printf("%.3f %d\n", predictions[i], i);
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                probs[index][j] = 0;
            }
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            //printf("%.3f %d\n", scale, obj_index);
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            if(masks){
                for(j = 0; j < l.coords - 4; ++j){
                    masks[index][j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            {
                float max = 0;
                for(j = 0; j < l.classes; ++j){
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                    //printf("%d %.3f\n", class_index, predictions[class_index]);
                    float prob = scale*predictions[class_index];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                    if(prob > max) {max = prob;
                      //                      printf("%.3f %.3f \n", prob, predictions[class_index]);
                    }
                }
                probs[index][l.classes] = max;
                //                printf("%.3f\n", probs[index][l.classes]);
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
    correct_region_boxes(boxes, l.w*l.h*l.n, w, h, netw, neth, relative);
}


void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            l.output[obj_index] = 0;
        }
    }
}

