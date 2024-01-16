use tch::{nn::{self, Module, ModuleT}, Tensor};

#[derive(Debug)]
pub struct DQNModelNN {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    res_block1: ResidualBlock,
    res_block2: ResidualBlock,
    res_block3: ResidualBlock,
    res_block4: ResidualBlock,
    res_block5: ResidualBlock,
    res_block6: ResidualBlock,
    res_block7: ResidualBlock,
    res_block8: ResidualBlock,
    res_block9: ResidualBlock,
    res_block10: ResidualBlock,
    fc1: nn::Linear,
}

#[derive(Debug)]
pub struct ResidualBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    adjust_conv: nn::Conv2D,
    in_channels: i64,
    out_channels: i64
}

impl ResidualBlock {
    fn new(vs: &nn::Path, in_channels: i64, out_channels: i64) -> ResidualBlock {
        let conv1 = nn::conv2d(vs, in_channels, out_channels, 1, Default::default());
        let bn1 = nn::batch_norm2d(vs, out_channels, tch::nn::BatchNormConfig { ..Default::default() });
        let conv2 = nn::conv2d(vs, out_channels, out_channels, 1, Default::default());
        let bn2 = nn::batch_norm2d(vs, out_channels, tch::nn::BatchNormConfig { ..Default::default() });
        let adjust_conv = nn::conv2d(vs, in_channels, out_channels, 1, Default::default());

        ResidualBlock {
            conv1,
            bn1,
            conv2,
            bn2,
            adjust_conv,
            in_channels,
            out_channels
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let residual = if self.in_channels == self.out_channels {
            x.shallow_clone()
        } else {
            self.adjust_conv.forward(x)
        };
        //let residual = x.shallow_clone();
        let conv1_out = self.conv1.forward(&x);
        let bn1_out = self.bn1.forward_t(&conv1_out, true).relu();
        
        let conv2_out = self.conv2.forward(&bn1_out);
        let bn2_out = self.bn2.forward_t(&conv2_out, true);
        
        bn2_out + residual
    }
}

impl DQNModelNN {
    pub fn new(vs: &nn::Path) -> DQNModelNN {
        let conv1 = nn::conv2d(vs, 1, 72, 3, Default::default());
        let bn1 = nn::batch_norm2d(vs, 72, tch::nn::BatchNormConfig { ..Default::default() });

        let res_block1 = ResidualBlock::new(vs, 72, 128);
        let res_block2 = ResidualBlock::new(vs, 128, 256);
        let res_block3 = ResidualBlock::new(vs, 256, 256);
        let res_block4 = ResidualBlock::new(vs, 256, 256);
        let res_block5 = ResidualBlock::new(vs, 256, 256);
        let res_block6 = ResidualBlock::new(vs, 256, 256);
        let res_block7 = ResidualBlock::new(vs, 256, 256);
        let res_block8 = ResidualBlock::new(vs, 256, 256);
        let res_block9 = ResidualBlock::new(vs, 256, 256);
        let res_block10 = ResidualBlock::new(vs, 256, 128);
        let fc1 = nn::linear(vs, 1152, 128, Default::default());

        DQNModelNN {
            conv1,
            bn1,
            res_block1,
            res_block2,
            res_block3,
            res_block4,
            res_block5,
            res_block6,
            res_block7,
            res_block8,
            res_block9,
            res_block10,
            fc1
        }
    }
}

impl nn::Module for DQNModelNN {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.to_kind(tch::Kind::Float).view([-1, 1, 9, 8]).to_device(tch::Device::Mps);

        let conv1_out = self.conv1.forward(&xs).max_pool2d_default(2);       
        let bn1_out = self.bn1.forward_t(&conv1_out, true).relu();

        let res1_out = self.res_block1.forward(&bn1_out).relu();
        let res2_out = self.res_block2.forward(&res1_out).relu();
        let res3_out = self.res_block3.forward(&res2_out).relu();
        let res4_out = self.res_block4.forward(&res3_out).relu();
        let res5_out = self.res_block5.forward(&res4_out).relu();
        let res6_out = self.res_block6.forward(&res5_out).relu();
        let res7_out = self.res_block7.forward(&res6_out).relu();
        let res8_out = self.res_block8.forward(&res7_out).relu();
        let res9_out = self.res_block9.forward(&res8_out).relu();
        let res10_out = self.res_block10.forward(&res9_out).relu();

        let [batch_size, num_channels, altura, ancho]: [i64; 4] = res10_out.size4().unwrap().into();
        let flattened = res10_out.view([batch_size, num_channels * altura * ancho]);

        let fc1_out = self.fc1.forward(&flattened);

        fc1_out

    }
}
