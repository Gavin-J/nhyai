<template>
	<div class="businessLicence">
		<el-row class="loading_con">
			<el-col :xs={span:24} :sm={span:11,offset:1} :md={span:10,offset:2} :lg={span:9,offset:3} :xl={span:8,offset:4}>
				<div class="image_outer">
					<div class="outer_add">
						<span class="original_style">原始图片</span>
						<img class="show_add_image" :src="dialogImageUrl">
					</div>
					<div class="upload_outer">
						<div class="local_upload" v-if="!isCheck">
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
				<div class="show_json_outer">
					<span class="original_style">识别结果</span>
					<div id="show_json" v-show="showJson.license_type">
						<p>证件类型：{{showJson.license_type}}</p>
						<p>统一社会信用代码：{{showJson.business_id}}</p>
						<p>名称：{{showJson.business_name}}</p>
						<p>类型：{{showJson.business_type}}</p>
						<p>公司地址：{{showJson.address}}</p>
						<p>法定代表人：{{showJson.operator}}</p>
						<p>注册资本：{{showJson.registered_capital}}</p>
						<p>注册日期：{{showJson.register_date}}</p>
						<p>经营期限：{{showJson.business_term}}</p>
						<p>经营范围：{{showJson.scope}}</p>
					</div>
				</div>
			</el-col>
		</el-row>
	</div>

</template>

<script>
    export default {
        data() {
            return {
                dialogImageUrl: require("../../assets/image/business_licence_sample.jpg"),
                dialogVisible: false,
                jsonDemo:'{"license_type":"营业执照","business_id":"91469027MA5RC58J61","business_name":"海南南天云网络科技有限公司","business_type":"有限责任公司","address":"海南省老城高新技术产业示范区海南生态软件园","operator":"王天云","registered_capital":"壹仟万元人民币","register_date":"2015年11月03日","business_term":"2015年11月03日至长期","scope":"从事互联网技术及计算机领域内的技术开发、技术转让、技术咨询、技术服务,计算机系统集成,网络工程。(一般经营项目一般经营项目自主经营，许可经营项目凭相关许可证或者批准文件经营）（依法须经批准的项目，经相关部门批准后方可开展经营活动。）"}',
                buttonWord:"开始检测",
                imageName:"",
                showPercent:"概率：1.75%",
                isForce:false,
                imageRight:false,
                imageIsBig:false,
                isCheck:false,
                activeName: 'first',
                showJson :{},
                options:{background:"rgba(0, 0, 0, 0.3)",fullscreen:false,target:document.querySelector(".outer_add")}

            };
        },
        mounted:function () {
            var that = this;
            this.isCheck = true;
            var jdata = JSON.stringify(JSON.parse(this.jsonDemo), null, 4);
            var loading = this.$loading({fullscreen:false,target:document.querySelector(".outer_add")});
            this.intervalid1 = setTimeout(() => {
                this.showJson = JSON.parse(this.jsonDemo);
                clearInterval(this.intervalid1);
                this.isCheck = false;
                this.dialogImageUrl=this.imgApi+"/media/photos/c21440d19d2a4fd78b8251dc66de372f_drawed.jpg";
                setTimeout(()=>{
                    loading.close();
                },1000)
            }, 2000)
        },
        methods: {
            uploadImage(e){
                let loading = this.$loading({fullscreen:false,target:document.querySelector(".outer_add")});
                this.isCheck = true;
                this.imageRight = false;
                this.showJson= [];
                var formData = new FormData();
                formData.append('image', $('#datafile')[0].files[0]);
                formData.append('system_id', 1);
                formData.append('channel_id', 9);
                $.ajax({
                    url: this.api+"/api/v1/ocr/get_business_licence_ocr/",
                    type: "post",
                    data: formData,
//                    headers: {'Authorization': 'Token mytoken'},
                    cache: false,
                    contentType: false,
                    processData: false,
                    success:(response)=>{
                        loading.close();
                        this.isCheck = false;
//                        this.uploadInfo(response);
                        console.log(response)
                        this.showJson = response.data;
                        this.dialogImageUrl = response.draw_url;
                    },
                    error:(error)=>{
                        this.$message.error('上传失败，请重新上传！');
                        loading.close();
                        this.isCheck= false;
                    }
                });
                e.preventDefault();
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
                    this.uploadImage(e);
                }
            },
        }
    }
</script>

<style scoped>
	.show_json_outer{height: 350px;overflow-y:scroll;border: 1px solid #e2ecfc;}

	.original_style{position: absolute;font-size: 14px;color: #316dff;height: 30px;line-height: 30px;background-color: #e3ecfb;display: inline-block;padding: 0 35px 0 25px;-webkit-clip-path: polygon(0% 0%, 100% 0%, 90% 100%, 0% 100%);}
	.image_outer{margin-right: 10px;}
	.show_add_image{max-height: 100%;max-width: 100%;vertical-align: middle;}
	.outer_add{height:350px;overflow: hidden;border: 1px solid #e2ecfc;vertical-align: middle;text-align: center;line-height: 350px;}
	.upload_outer{display: flex;margin-top: 20px;}
	.top_suggest{color: #999999;font-size: 14px;line-height: 40px;height: 30px;}
	.init_url_style{flex: 1;height: 43px;line-height: 43px;border: 1px solid #E2ECFC;font-size: 15px;padding-left: 10px;background-color: #fafcfe;}
	.init_url_style:hover{border: 1px solid #C0C4CC;border-right: none;}
	.init_url_style:focus{border: 1px solid #409EFF;border-right: none;}
	.check_style{display:inline-block;height: 41px;line-height: 41px;font-size: 16px;color: #316dff;border: 2px solid #316dff;width: 100px;text-align: center;cursor:pointer;background-color: #fafcfe;}
	.check_style:hover{background-color: #316DFF;color: white;}
	.is_check{display:inline-block;height: 43px;line-height: 43px;font-size: 16px;background-color: #f5f5f5;color:#666666;border: 1px solid #dddddd;padding: 0 30px;text-align: center;}
	.local_upload{height: 45px;line-height: 45px;font-size: 16px;}

	/*.local_upload:after{content: "或";margin: 0 15px;}*/
	.inputfile{z-index: -11111;width: 0px;height:1px;opacity: 0;position: absolute;}
	.local_upload label{display:inline-block;height: 43px;line-height: 43px;font-size: 16px;background-color: #316DFF;color:white;border: 1px solid #316DFF;padding: 0 30px;text-align: center;cursor: pointer;}
	.local_upload label:hover{background-color: #6087F7;color: white;}
	.show_input_outer{display: flex;flex: 1;}
	#show_json{margin: 50px auto;padding: 10px 30px;}
	#show_json p{line-height: 30px;}

	.advantage_product span{display: inline-block;padding: 10px;}
	.show_word{width: 192px;height: 148px;  display: inline-block;  text-align: center;  font-size: 22px;  font-style: normal;  color: #fff;  box-sizing: border-box;  padding-top: 58px;}
	.normal_result_back{background-image: url("../../assets/image/2.png"); background-position: 0 0;}
	.danger_result_back{background-image: url("../../assets/image/2.png"); background-position: -196px 0;}
	.probability_number{height: 30px;line-height: 30px;text-align: center;color: #fff;font-size: 16px;margin-top: 10px;}
</style>