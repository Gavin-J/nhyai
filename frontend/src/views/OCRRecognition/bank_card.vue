<template>
	<div class="bankCard">
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
							<input id="datafile" name="datafile" type="file" accept="image/*" class="inputfile" @change="changeImage($event)">
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
					<div id="show_json" v-show="showJson.bank_name">
						<p>银行信息：{{showJson.bank_name|noCheck}}</p>
						<p>卡号：{{showJson.bank_cardno|noCheck}}</p>
						<p>有效期：{{showJson.expiry_date|noCheck}}</p>
						<p>银行卡类型：{{showJson.card_type|noCheck}}</p>
						<p>卡名称：{{showJson.card_name|noCheck}}</p>
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
                dialogImageUrl: require("../../assets/image/back_sample.png"),
                dialogVisible: false,
                jsonDemo:'{"bank_name":"中国工商银行","bank_cardno":"6225970010000000","expiry_date":"2022年07月","card_type":"信用卡","card_name":"未识别"}',
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
            this.isCheck = true;
            var jdata = JSON.stringify(JSON.parse(this.jsonDemo), null, 4);
            let loading = this.$loading({fullscreen:false,target:document.querySelector(".outer_add")});
            this.intervalid1 = setTimeout(() => {
                this.dialogImageUrl = this.imgApi +'/media/photos/3713fc4cb98548cbaf7b56d9d8cc2e7a_drawed.png';
                this.showJson = JSON.parse(this.jsonDemo);
                clearInterval(this.intervalid1);
                this.isCheck = false;
                setTimeout(()=>{
                    loading.close();
                },1000)
            }, 2000);
        },
        methods: {
            uploadImage(e){
                let loading = this.$loading({fullscreen:false,target:document.querySelector(".outer_add")});
                this.isCheck= true;
                this.imageRight = false;
                var formData = new FormData();
                formData.append('image', $('#datafile')[0].files[0]);
                formData.append('system_id', 1);
                formData.append('channel_id', 10);
                $.ajax({
                    url: this.api+"/api/v1/ocr/get_bank_card_ocr/",
                    type: "post",
                    data: formData,
//                    headers: {'Authorization': 'Token mytoken'},
                    cache: false,
                    contentType: false,
                    processData: false,
                    success:(response)=>{
                        loading.close();
                        console.log(response);
                        this.isCheck= false;
                        this.showJson  = response.data;
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
	.show_add_image{max-height: 100%;max-width: 100%;vertical-align: middle;}
	.original_style{position: absolute;font-size: 14px;color: #316dff;height: 30px;line-height: 30px;background-color: #e3ecfb;display: inline-block;padding: 0 35px 0 25px;-webkit-clip-path: polygon(0% 0%, 100% 0%, 90% 100%, 0% 100%);}
	.image_outer{margin-right: 10px;}
	.outer_add{height:350px;overflow: hidden;border: 1px solid #e2ecfc;vertical-align: middle;text-align: center;line-height: 350px;}
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
	.local_upload label{display:inline-block;height: 43px;line-height: 43px;font-size: 16px;background-color: #316DFF;color:white;border: 1px solid #316DFF;padding: 0 30px;text-align: center;cursor: pointer;}
	.local_upload label:hover{background-color: #6087F7;color: white;}
	.show_input_outer{display: flex;flex: 1;}

	#show_json{margin: 50px auto;padding: 10px 30px;}
	#show_json p{height: 30px;line-height: 30px;}
	.inputfile{z-index: -11111;width: 0;height:1px;opacity: 0;position: absolute;}
	.suggest{color: #b2b2b2;font-size: 14px;min-height: 30px;margin:8px 0;}
	.show_result_outer{width: 100%;height: 100%;position: absolute;top: 0;z-index: 1;text-align: center; background-color: rgba(0,0,0,.4);}
	.show_result_outer:before{content: '';position: absolute;left: 0;top: 0;width: 100%;height: 100%;}
	.show_result{ z-index: 3;display: inline-block;vertical-align: middle;margin-top: 15%}

	.advantage_product span{display: inline-block;padding: 10px;}
	.show_word{width: 192px;height: 148px;  display: inline-block;  text-align: center;  font-size: 22px;  font-style: normal;  color: #fff;  box-sizing: border-box;  padding-top: 58px;}
	.normal_result_back{background-image: url("../../assets/image/2.png"); background-position: 0 0;}
	.danger_result_back{background-image: url("../../assets/image/2.png"); background-position: -196px 0;}
	.probability_number{height: 30px;line-height: 30px;text-align: center;color: #fff;font-size: 16px;margin-top: 10px;}
</style>