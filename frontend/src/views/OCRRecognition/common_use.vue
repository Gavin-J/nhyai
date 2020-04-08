<template>
	<div class="commonUse">
		<el-row class="loading_con">
			<el-col :xs={span:24} :sm={span:11,offset:1} :md={span:10,offset:2} :lg={span:9,offset:3} :xl={span:8,offset:4}>
				<div class="image_outer">
					<div class="outer_add">
						<span class="original_style original_title_style">原始图片</span>
						<img class="show_add_image" :src="dialogImageUrl" id="img" v-show="!isResult">
						<canvas id="myCanvas" class="show_add_image"></canvas>
					</div>
					<div class="upload_outer">
						<div class="local_upload" v-if="!isCheck">
							<!--<p>本地上传</p>-->
							<input id="datafile" name="datafile" type="file" accept="image/*" class="inputfile" @change="changeImage($event)">
							<label for="datafile">本地上传</label>
						</div>
						<div class="local_upload" v-else>
							<p class="is_check">正在检测</p>
						</div>
						<!--<div class="show_input_outer">
							<input type="text" class="init_url_style" readonly  placeholder="请输入网络图片URL">
							<p class="check_style_hidden" @click="urlCheck">检测</p>
						</div>-->
					</div>
					<p class="top_suggest">提示：图片大小不超过20M，请保证需要识别部分为图片主体部分</p>
				</div>
			</el-col>
			<el-col :xs={span:24} :sm={span:11} :md="10" :lg="9" :xl="8">
				<div class="show_json_outer">
					<span class="original_style">识别结果</span>
					<div id="show_common_json">
					</div>
				</div>
			</el-col>
		</el-row>
	</div>

</template>

<script>
    import {canvasBox,clearCanvas} from '../../store/common'
    export default {
        data() {
            return {
                dialogImageUrl: require("../../assets/image/common_sample.png"),
                dialogVisible: false,
                jsonDemo:'{"name":"艾米","sex":"女","nation":"汉","birthday":"1986年4月23日","address":"上海徐汇区田林路397号腾云大厦6F","idNumber":"310104198604230289"}',
                buttonWord:"开始检测",
                imageName:"",
                showPercent:"概率：1.75%",
                isForce:false,
                imageRight:false,
                imageIsBig:false,
                activeName: 'first',
                showJson :{},
                options:{fullscreen:false,target:document.querySelector(".outer_add")},
				isCheck:false,
                isResult: false,
                currentBox: [[296.1687175081781,82.0341836846865,491.09910474147654,82.96645675430968,490.8312824918219,138.9658163153135,295.90089525852346,138.03354324569034],[308.1,159.0,489.9,159.0,489.9,203.0,308.1,203.0],[30.435277228695895,274.002590936372,743.4963570494341,272.99745724789864,743.5647227713041,321.497409063628,30.503642950565904,322.50254275210136],[29.302396170131033,370.00284339279887,769.6324021020525,368.99720089123525,769.697603829869,416.99715660720113,29.367597897947462,418.00279910876475]]

            };
        },
        mounted:function () {
            var that = this;
            this.isCheck= true;
            var jdata = JSON.stringify(JSON.parse(this.jsonDemo), null, 4);
            var loading = this.$loading({fullscreen:false,target:document.querySelector(".outer_add")});
            this.intervalid1 = setTimeout(() => {
                document.getElementById('show_common_json').innerHTML= "赠汪伦\n" +'<br/>'+
                    "【唐】李白\n"+'<br/>'+
                    "李白乘舟将欲行,忽闻岸上踏歌声。"+'<br/>' +
                    "桃花潭水深千尺,不及汪伦送我情。";
                clearInterval(this.intervalid1);
                this.isCheck= false;
                this.plotBox(this.currentBox,this.dialogImageUrl,loading);
            }, 2000);

        },
        methods: {
            urlCheck(){
                this.$message.error('该功能尚未开通！')
            },
            uploadImage(e){
                this.isCheck= true;
                var loading = this.$loading({fullscreen:false,target:document.querySelector(".outer_add")});
                var formData = new FormData();
                formData.append('image', $('#datafile')[0].files[0]);
                formData.append('system_id', 1);
                formData.append('channel_id', 8);
                document.getElementById('show_common_json').innerHTML="";
                $.ajax({
                    url: this.api+"/api/v1/ocr/get_general_ocr/",
                    type: "post",
                    data: formData,
//                    headers: {'Authorization': 'Token mytoken'},
                    cache: false,
                    contentType: false,
                    processData: false,
                    success:(response)=>{
						console.log(response);

						var text =''
                        response.data.forEach((res)=>{
                            text= text+res+'<br/>'
						});
                        if(response.data.length==0){
                            text = "未识别到内容，请重新识别！"
                        }else{
//                            this.plotBox(response.box,response.image,loading);
						}
						this.dialogImageUrl = response.draw_url;
						document.getElementById('show_common_json').innerHTML= text;
                        loading.close();
                        this.isCheck= false;

                    },
					error:(error)=>{
                        this.$message.error('上传失败，请重新上传！');
                        loading.close();
                        this.isCheck= false;
					}
                });
                e.preventDefault();
            },
            plotBox(boxes,src,loading){
                $('#myCanvas').css('display','inline-block');
                canvasBox(boxes,src,document.getElementById("img"),document.getElementById("myCanvas"),()=>{
                    this.isResult = true;
                    this.isLoading = false;
                    loading.close();
                });
            },
            changeImage(e){
                this.imageIsBig = false;
                this.imageRight = false;
                const file = e.target.files[0];
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
                    clearCanvas(document.getElementById("myCanvas"));
                    this.uploadImage(e);
                }
            },
        }
    }
</script>

<style scoped>
	.show_json_outer{height: 350px;overflow-y:scroll;border: 1px solid #e2ecfc;}
	.show_add_image{max-width: 100%;max-height:100%;vertical-align: middle;}
	.original_style{position: absolute;font-size: 14px;color: #316dff;height: 30px;line-height: 30px;background-color: #e3ecfb;display: inline-block;padding: 0 35px 0 25px;-webkit-clip-path: polygon(0% 0%, 100% 0%, 90% 100%, 0% 100%);}
	.original_title_style{top: 0;left: 0;}
	.image_outer{margin-right: 10px;}
	.outer_add{height:350px;line-height:350px;position: relative;overflow: hidden;border: 1px solid #e2ecfc;vertical-align: middle;text-align: center;}
	.upload_outer{display: flex;margin-top: 20px;}
	.top_suggest{color: #999999;font-size: 14px;line-height: 40px;height: 30px;}
	.init_url_style{flex: 1;height: 43px;line-height: 43px;border: 1px solid #E2ECFC;font-size: 15px;padding-left: 10px;background-color: #fafcfe;}
	/*.init_url_style:hover{border: 1px solid #C0C4CC;border-right: none;}*/
	/*.init_url_style:focus{border: 1px solid #409EFF;border-right: none;}*/
	.check_style{display:inline-block;height: 41px;line-height: 41px;font-size: 16px;color: #316dff;border: 2px solid #316dff;width: 100px;text-align: center;cursor:pointer;background-color: #fafcfe;}
	.check_style_hidden{display:inline-block;height: 41px;line-height: 41px;font-size: 16px;color: #666666;border: 2px solid #f5f5f5;
		width: 100px;text-align: center;cursor:pointer;background-color: #f5f5f5}
	.check_style:hover{background-color: #316DFF;color: white;}
	.local_upload{height: 45px;line-height: 45px;font-size: 16px;}
	/*.local_upload:after{content: "或";margin: 0 15px;}*/
	.inputfile{z-index: -11111;width: 0px;height:1px;opacity: 0;position: absolute;}
	.local_upload label{display:inline-block;height: 43px;line-height: 43px;font-size: 16px;background-color: #316DFF;color:white;border: 1px solid #316DFF;padding: 0 30px;text-align: center;cursor: pointer;}
	.local_upload label:hover{background-color: #6087F7;color: white}
	.is_check{display:inline-block;height: 43px;line-height: 43px;font-size: 16px;background-color: #f5f5f5;color:#666666;border: 1px solid #dddddd;padding: 0 30px;text-align: center;}
	.show_input_outer{display: flex;flex: 1;}
	#show_common_json{margin: 50px auto;padding: 10px 30px;word-break:break-all;line-height: 30px;}
	#show_common_json p{height: 30px;line-height: 30px;}

	.advantage_product span{display: inline-block;padding: 10px;}
	.show_word{width: 192px;height: 148px;  display: inline-block;  text-align: center;  font-size: 22px;  font-style: normal;  color: #fff;  box-sizing: border-box;  padding-top: 58px;}
	.normal_result_back{background-image: url("../../assets/image/2.png"); background-position: 0 0;}
	.danger_result_back{background-image: url("../../assets/image/2.png"); background-position: -196px 0;}
	.probability_number{height: 30px;line-height: 30px;text-align: center;color: #fff;font-size: 16px;margin-top: 10px;}
</style>