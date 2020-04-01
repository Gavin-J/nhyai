<template>
	<div class="handwrite">
		<el-row class="loading_con">
			<el-col :xs={span:24} :sm={span:11,offset:1} :md={span:10,offset:2} :lg={span:9,offset:3} :xl={span:8,offset:4}>
				<div class="image_outer">
					<span class="original_style">原始图片</span>
					<div class="outer_add" v-loading="isLoading">

						<img class="show_add_image" :src="dialogImageUrl" id="img" v-show="!isResult">
						<canvas id="myCanvas" class="show_add_image"></canvas>
					</div>
					<div class="upload_outer">
						<div class="local_upload" v-if="!isLoading">
							<!--<p>本地上传</p>-->
							<input id="datafile" name="datafile" type="file" class="inputfile" @change="changeImage($event)">
							<label for="datafile">本地上传</label>
						</div>
						<div class="local_upload" v-else>
							<p class="is_check">正在检测</p>
						</div>
						<!--<div class="show_input_outer">
							<input type="text" class="init_url_style" placeholder="请输入网络图片URL">
							<p class="check_style">检测</p>
						</div>-->
					</div>
					<p class="top_suggest">提示：图片大小不超过20M，请保证需要识别部分为图片主体部分</p>
				</div>
			</el-col>
			<el-col :xs={span:24} :sm={span:11} :md="10" :lg="9" :xl="8">
				<div class="show_json_outer"  >
					<span class="original_style">识别结果</span>
					<div id="show_json" v-show="showJson">
						<p v-for="item in showJson">{{item}}</p>
					</div>
				</div>
			</el-col>
		</el-row>
	</div>

</template>

<script>
	import {canvasBox} from '../../store/common'
    export default {
        data() {
            return {
                dialogImageUrl: require("../../assets/image/hand_writer_sample.png"),
                dialogVisible: false,//
                jsonDemo:'["人生就像一盒巧克力","你永远都不知道下一块会是什么味道","Liffe was like a box of chocalates.You never know","what you\'re going to get.","《阿甘正传》"]',
                buttonWord:"开始检测",
                imageName:"",
                showPercent:"概率：1.75%",
                isForce:false,
                imageRight:false,
                imageIsBig:false,
                activeName: 'first',
                showJson :{},
                options:{background:"rgba(0, 0, 0, 0.3)",fullscreen:false,target:document.querySelector(".outer_add")},
                currentImage:1,
                isLoading:false,
				clickFirst:0,
                isResult:false
            };
        },
        mounted:function () {
            this.loadDate();
        },
        methods: {
            uploadImage(e){
                this.isLoading = true;
                this.showJson = {};
                this.imageRight = false;
                var formData = new FormData();
                formData.append('image', $('#datafile')[0].files[0]);
                formData.append('system_id', 1);
                formData.append('channel_id', 11);
                $.ajax({
                    url: this.api+"/api/v1/ocr/get_handwritten_ocr/",
                    type: "post",
                    data: formData,
//                    headers: {'Authorization': 'Token mytoken'},
                    cache: false,
                    contentType: false,
                    processData: false,
                    success:(response)=>{
                        this.isLoading = false;
                        console.log(response);
                        this.showJson = response.data.handwritten_content;
                        $('#myCanvas').css('display','inline-block');
                        this.plotBox(response.data.box,response.image)

                    },
                    error:(error)=>{
                        this.$message.error('上传失败，请重新上传！');
                        this.isLoading = false;
                        this.isCheck= false;
                    }
                });
                e.preventDefault();
            },
            plotBox(boxes,src){
                this.isResult = true;
                canvasBox(boxes,src,document.getElementById("img"),document.getElementById("myCanvas"))
            },
            changeImage(e){
                this.imageIsBig = false;
                this.imageRight = false;
                const file = e.target.files[0];
                console.log(file)
                const reader = new FileReader();
                const that = this;
                reader.readAsDataURL(file);
                reader.onload = function() {
                    that.dialogImageUrl = this.result;
                };
                let size=file.size;//文件的大小，判断图片的大小
                if(size>1048576*20){
                    this.$message.error('请上传小于20M的图片！');
                }else {
                    this.imageRight = true;
                    this.isResult = false;
                    $('#myCanvas').css('display','none');
                    this.uploadImage(e);
                }
            },
            loadDate() {
                this.isLoading = true;
                $(".show_sm_image").attr("disabled",true).css("pointer-events","none");
                console.log(this.clickFirst)
                if(this.clickFirst===0){
                    this.isLoading = true;
                    this.clickFirst+=1;
                    this.showJson= {};
                    var intervalid1 = setTimeout(() => {
                        this.showJson = JSON.parse(this.jsonDemo);
                        clearInterval(intervalid1);
                        this.isLoading = false;
                        $(".show_sm_image").attr("disabled",false).css("pointer-events","auto");
                        this.clickFirst = 0;
                    }, 4000)
				}

            }
        }
    }
</script>

<style scoped>
	.show_json_outer{height: 350px;overflow-y:scroll;border: 1px solid #e2ecfc;}
	.original_style{position: absolute;font-size: 14px;color: #316dff;height: 30px;line-height: 30px;background-color: #e3ecfb;display: inline-block;
		padding: 0 35px 0 25px;-webkit-clip-path: polygon(0% 0%, 100% 0%, 90% 100%, 0% 100%);z-index: 1;}
	.image_outer{margin-right: 10px;}
	.show_add_image{max-height: 100%;max-width: 100%;vertical-align: middle;}
	.outer_add{position:relative; height:350px;overflow: hidden;border: 1px solid #e2ecfc;vertical-align: middle;text-align: center;line-height: 350px;}
	.upload_outer{display: flex;margin-top: 20px;}
	.top_suggest{color: #999999;font-size: 14px;line-height: 40px;height: 30px;}
	.init_url_style{flex: 1;height: 43px;line-height: 43px;border: 1px solid #E2ECFC;font-size: 15px;padding-left: 10px;background-color: #fafcfe;}
	.init_url_style:hover{border: 1px solid #C0C4CC;border-right: none;}
	.init_url_style:focus{border: 1px solid #409EFF;border-right: none;}
	.check_style{display:inline-block;height: 41px;line-height: 41px;font-size: 16px;color: #316dff;border: 2px solid #316dff;width: 100px;text-align: center;cursor:pointer;background-color: #fafcfe;}
	.check_style:hover{background-color: #316DFF;color: white;}
	.local_upload{height: 45px;line-height: 45px;font-size: 16px;}
	/*.local_upload:after{content: "或";margin: 0 15px;}*/
	.is_check{display:inline-block;height: 43px;line-height: 43px;font-size: 16px;background-color: #f5f5f5;color:#666666;border: 1px solid #dddddd;padding: 0 30px;text-align: center;}
	.inputfile{z-index: -11111;width: 0px;height:1px;opacity: 0;position: absolute;}
	.local_upload label{display:inline-block;height: 43px;line-height: 43px;font-size: 16px;background-color: #316DFF;color:white;border: 1px solid #316DFF;padding: 0 30px;text-align: center;cursor: pointer;}
	.local_upload label:hover{background-color: #6087F7;color: white;}
	.show_input_outer{display: flex;flex: 1;}
	#show_json{margin: 50px auto;padding: 10px 30px;}
	#show_json p{height: 30px;line-height: 30px;}

	.advantage_product span{display: inline-block;padding: 10px;}
</style>