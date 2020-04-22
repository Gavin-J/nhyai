<template>
	<div id="sindex">
		<Navigation></Navigation>
		<div class="top_contain">
			<el-row>
				<el-row>
					<el-col :xl={span:24}>
						<div class="banner_outer ai-common-banner">
							<!--<img src="../assets/image/write/write_banner.png" alt="">-->
							<div class="describe_outer_banner">
								<p class="ell">AI智能审查</p>
								<p class="ell-rows-4 ">基于深度学习的图像识别算法，海量大数据样本，准确识别图片和视频中的涉黄、
									涉暴涉恐、政治敏感、民谣等内容，也能从美观和清晰等维度对图像进行筛选，
									快速精准，解放审核人力</p>
								<span>
									<img src="../assets/image/sindex/sindex_banner_mark1.png" alt="">
									<img src="../assets/image/sindex/sindex_banner_mark2.png" alt="">
									<img src="../assets/image/sindex/sindex_banner_mark3.png" alt="">
									<img src="../assets/image/sindex/sindex_banner_mark4.png" alt="">

								</span>
							</div>
							<span class="tech-banner-box"></span>
						</div>
						<!--<img src="../assets/image/image_banner.png" alt="">-->
					</el-col>
				</el-row>
			</el-row>
		</div>
		<div class="functional_experience" id="practice_title">
			<p class="title">功能体验</p>
			<div class="current_width_style clearfix">
				<div class="show_input_outer fl">
					<input type="text" class="init_url_style" id="contentUrl"  placeholder="请输入网络图片URL" v-model="fileUrl">
					<p class="check_style" @click="urlCheck()">检测</p>
					<!--<input type="file" id="url_input" class="inputfile" @change="changeImage($event)" multiple>-->
					<!--<label for="url_input" class="init_url_style">请选择本地文件上传</label>-->
					<!--<p class="check_style_hidden" @click="showInputValue">检测</p>-->
				</div>
				<div class="local_upload fl">
					<!--<p>本地上传</p>-->
					<input id="datafile" name="datafile" type="file" class="inputfile" @change="changeImage($event)">
					<label for="datafile" v-if="!isUploading" class="btn_upload" @click="fileUrl =''">本地上传</label>
					<label v-else class="btn_uploading">本地上传</label>
				</div>
			</div>

			<p class="top_suggest current_width_style">支持图片格式：PNG、JPG、JPEG，大小限制<20M。支持视频格式：mp4，大小限制<20M；支持音频格式：wav；支持文本格式：txt，大小限制<10M</p>

			<!--图片评审-->
			<ImageCheck v-show="checkType ===1" :file="imageFile" ref="imageCheck"></ImageCheck>

			<!--视频评审-->
			<VideoCheck :stopVideo="stopVideo" v-show="checkType ===2" ref="videoCheck"></VideoCheck>

			<!--音频评审-->
			<AudioCheck :stopAudio="stopAudio" v-show="checkType ===3" ref="audioCheck"></AudioCheck>
			<TextCheck v-show="checkType ===4" ref="textCheck"></TextCheck>

		</div>
		<div class="functional_experience">
			<p class="title">历史记录</p>
			<div style="width: 1200px;margin: 0 auto;">
				<div class="show_search_con">
					<span>上传时间</span>
					<el-date-picker
						v-model="beginDate"
						type="date"
						placeholder="起始日期"
						:picker-options="pickerOptionsStart"
						@change="changeEnd">
					</el-date-picker>
					<span>-</span>
					<el-date-picker
						v-model="endDate"
						type="date"
						placeholder="截止日期"
						:picker-options="pickerOptionsEnd"
						@change="changeStart">
					</el-date-picker>
					<input type="text" class="search_input" v-model="searchContent">
					<span class="search_btn" @click="searchHistory">查询</span>
				</div>
				<table border="0" width="100%">
					<tr class="show_history_title">
						<th>文件名</th>
						<th>审查结果</th>
						<th>敏感类型</th>
						<th>敏感系数</th>
						<th>文本</th>
						<th>上传时间</th>
						<th>预览</th>
					</tr>
					<tr class="show_history_con" v-for="(item ,index) in historyInfo">

						<td v-if="item.file_type==5" class="ell">手动输入，无文件名</td>
						<td v-else class="ell">{{item.file_name}}</td>
						<td class="green_color" v-if="item.max_sensitivity_level==0">合规</td>
						<td class="orange_color" v-else-if="item.max_sensitivity_level==1">疑似违规</td>
						<td class="red_color" v-else-if="item.max_sensitivity_level==2">违规</td>
						<td class="green_color" v-else>/</td>
						<td class="ell" v-if="item.max_sensitivity_type=='violence'">暴恐</td>
						<td class="ell" v-else-if="item.max_sensitivity_type=='porn'">涉黄</td>
						<td class="ell" v-else-if="item.max_sensitivity_type=='violence_porn'">涉黄、暴恐</td>
						<td class="ell" v-else-if="item.max_sensitivity_type=='text'">文本识别</td>
						<!--<td class="ell" v-else-if="item.max_sensitivity_type=='ocr'">OCR识别</td>-->
						<td class="ell" v-else-if="item.channel_id=='5'">身份证识别</td>
						<td class="ell" v-else-if="item.channel_id=='6'">行驶证识别</td>
						<td class="ell" v-else-if="item.channel_id=='7'">驾驶证识别</td>
						<td class="ell" v-else-if="item.channel_id=='8'">通用识别</td>
						<td class="ell" v-else-if="item.channel_id=='9'">营业执照识别</td>
						<td class="ell" v-else-if="item.channel_id=='10'">银行卡识别</td>
						<td class="ell" v-else-if="item.channel_id=='11'">手写体识别</td>
						<td class="ell" v-else-if="item.channel_id=='12'">车牌识别</td>
						<td class="ell" v-else-if="item.channel_id=='13'">名片识别</td>
						<td class="ell" v-else-if="item.max_sensitivity_type=='audio'">语音识别</td>
						<td class="ell" v-else-if="item.file_type=='5'">敏感字识别</td>
						<td class="ell" v-else-if="item.max_sensitivity_type=='-1'">未识别</td>
						<td v-if="item.max_sensitivity_type !='ocr'">{{item.max_sensitivity_percent}}%</td>
						<td class="green_color" v-else-if="item.channel_id=='4'">/</td>
						<td class="green_color" v-else>/</td>
						<td class="ell" v-if="item.app_text">{{item.content}}</td>
						<td class="ell" v-else-if="item.web_text">{{item.web_text}}</td>
						<td class="ell" v-else-if="item.channel_id ==5">详情请预览</td>
						<td class="ell" v-else-if="item.channel_id ==8">详情请预览</td>
						<td class="ell" v-else>无</td>
						<td class="ell">{{item.upload_time|momentDate}}</td>
						<td  @click="preLook(item)" v-if="item.porn_percent>50" class="disAllow">预览</td>
						<td  @click="preLook(item)" v-else>预览</td>
					</tr>
					<tr v-show="!historyInfo.length" class="show_history_center">暂时没有记录哦~</tr>

				</table>
				<div class="show_pagination">
					<el-pagination
						background
						:current-page="currentPager"
						:prev-text="prevText"
						:next-text="nextText"
						:page-size="10"
						@current-change="handleCurrentChange"
						layout="prev, pager, next"
						:total="this.count">
					</el-pagination>
				</div>
				<el-dialog
					:visible.sync="centerDialogVisible"
					width="920px"
					@close="closeDialog"
					center>
					<div class="show_video_outer" v-show="preLookInfo.channel_id !==4">
						<div class="image_con" v-show="fileSort ==1">
							<el-image v-if="preLookInfo.draw_url"
									  style="max-height: 440px;max-width:99%;object-fit: cover;vertical-align: middle!important;"
									  :src="preLookInfo.draw_url"
									  :preview-src-list="srcList">
								<div slot="placeholder" class="image-slot"><span class="dot"></span>
								</div>
							</el-image>
							<el-image v-else
									  style="max-height: 440px;max-width:99%;object-fit: cover;vertical-align: middle!important;"
									  :src="preLookInfo.file_url"
									  :preview-src-list="srcList">
								<div slot="placeholder" class="image-slot">
								</div>
							</el-image>
							<!--<img :src="preLookInfo.file_url">-->
							<span></span>
						</div>
						<div class="image_con" v-show="fileSort ==3">
							<div>
								<audio :src="preLookInfo.file_url" controls width="420"></audio>
							</div>
							<!--<p>{{preLookInfo.file_name}}</p>-->
							<!--<audioExample :theUrl="preLookInfo.file_url"></audioExample>-->

						</div>
						<div class="image_con" v-show="fileSort ==2">
							<video id="Video" class="video-js vjs-defalut-skin" controls preload="metadata"  :src="videoUrl.video_url">
								<source :src="videoUrl.video_url" type="video/mp4">
								<!--<source src="" type="video/mp4">-->
								您的浏览器不支持视频
							</video>
						</div>
					</div>

					<div class="pre_result_con">
						<table width="100%" cellspacing="0" cellpadding="0">
							<tr class="pre_item">
								<td class="pre_item_common">文件名称</td>
								<td v-if="preLookInfo.file_type==5" class="ell">手动输入，无文件名</td>
								<td v-else>{{preLookInfo.file_name}}</td>
							</tr>
							<tr class="pre_item">
								<td class="pre_item_common">上传时间</td>
								<td>{{preLookInfo.upload_time|momentDate}}</td>
							</tr>
							<tr class="pre_item" v-show="preLookInfo.max_sensitivity_type!='ocr'">
								<td class="pre_result_item">审查结果</td>
								<td v-if="preLookInfo.file_type ==1|preLookInfo.file_type ==2" class="type_outer">
									<div class="clearfix" v-if="preLookInfo.violence_sensitivity_level !=-1">
										<div class="result_outer result_outer_notop fl">
											<p >暴恐识别</p>
											<p class="green_style_name" v-if="preLookInfo.violence_sensitivity_level==0">合规</p>
											<p class="orange_style_name" v-else-if="preLookInfo.violence_sensitivity_level==1">疑似违规</p>
											<!--<p class="green_style_name" v-else-if="preLookInfo.violence_sensitivity_level==-1">未检测</p>-->
											<p class="red_style_name" v-else-if="preLookInfo.violence_sensitivity_level==2">违规</p>
											<p class="green_style_number" v-if="preLookInfo.violence_sensitivity_level==0">{{preLookInfo.violence_percent}}%</p>
											<p class="orange_style_number" v-else-if="preLookInfo.violence_sensitivity_level==1">{{preLookInfo.violence_percent}}%</p>
											<!--<p class="green_style_number" v-else-if="preLookInfo.violence_sensitivity_level==-1">/</p>-->
											<p class="red_style_number" v-else-if="preLookInfo.violence_sensitivity_level==2">{{preLookInfo.violence_percent}}%</p>
										</div>
										<!--<div class="result_outer result_outer_notop left_50 fl">
											<p>政治敏感识别</p>
											<p class="green_style_name">合规</p>
											<p class="green_style_number">12.56%</p>
										</div>-->
									</div>
									<div class="clearfix" v-if="preLookInfo.porn_sensitivity_level !=-1">
										<div class="result_outer fl">
											<p class="ell">色情识别</p>
											<p class="green_style_name" v-if="preLookInfo.porn_sensitivity_level==0">合规</p>
											<p class="orange_style_name" v-else-if="preLookInfo.porn_sensitivity_level==1">疑似违规</p>
											<!--<p class="green_style_name" v-else-if="preLookInfo.porn_sensitivity_level==-1">未检测</p>-->
											<p class="red_style_name" v-else-if="preLookInfo.porn_sensitivity_level==2">违规</p>
											<p class="green_style_number" v-if="preLookInfo.porn_sensitivity_level==0">{{preLookInfo.porn_percent}}%</p>
											<p class="orange_style_number" v-else-if="preLookInfo.porn_sensitivity_level==1">{{preLookInfo.porn_percent}}%</p>
											<!--<p class="green_style_number" v-else-if="preLookInfo.porn_sensitivity_level==-1">/</p>-->
											<p class="red_style_number" v-else-if="preLookInfo.porn_sensitivity_level==2">{{preLookInfo.porn_percent}}%</p>
										</div>
										<!--<div class="result_outer left_50 fl" >
											<p>公众人物识别</p>
											<p class="red_style_name">违规</p>
											<p class="red_style_number">90.35%</p>
										</div>-->
									</div>
									<!--<div class="clearfix">
										<div class="result_outer fl">
											<p>广告检测</p>
											<p class="red_style_name">违规</p>
											<p class="red_style_number">90.16%</p>
										</div>
									</div>-->
								</td>
								<td v-else-if="preLookInfo.file_type ==4">
									<div class="clearfix" v-if="videoUrl.sensitive_info">
										<div class="result_outer text_result_outer fl" v-for="item in videoUrl.sensitive_info.final_list">
											<p>{{item}}</p>
											<p class="red_style_name" >违规</p>
											<!--<p class="red_style_number">{{item}}</p>-->
										</div>
									</div>
								</td>
								<td v-else-if="preLookInfo.file_type ==3">
									<div class="clearfix" v-if="videoUrl">
										<div class="result_outer text_result_outer fl" v-for="item in videoUrl.final_list">
											<p>{{item}}</p>
											<p class="red_style_name" >违规</p>
											<!--<p class="red_style_number">{{item}}</p>-->
										</div>
										<p v-if="!videoUrl.final_list">无</p>
									</div>
								</td>
							</tr>
							<tr class="pre_item">
								<td class="pre_word_item">文本识别</td>
								<!--<td v-if="preLookInfo.web_text" id="web_text">{{preLookInfo.web_text}}</td>-->
								<td v-if="preLookInfo.file_type==1">{{videoUrl.text|noCheck}}</td>
								<td v-else-if="preLookInfo.file_type==2">{{videoUrl.text|noCheck}}</td>
								<td v-else-if="preLookInfo.file_type==3">{{audioWord|noCheck}}</td>
								<td v-else-if="preLookInfo.file_type==4">{{videoUrl.text|noCheck}}</td>
								<td v-else-if="preLookInfo.channel_id==2">{{videoUrl.text|noCheck}}</td>
								<td v-else-if="preLookInfo.channel_id==3">{{videoUrl.text|noCheck}}</td>
								<td v-else-if="preLookInfo.channel_id==4" id="web_text"></td>
								<td v-else-if="preLookInfo.channel_id==5">姓名：{{videoUrl.name|noCheck}}<br/>性别：{{videoUrl.sex|noCheck}}<br/>
									民族：{{videoUrl.nation|noCheck}}<br/>出生：{{videoUrl.birth|formatDate('yyyy年MM月dd日')}}<br/>地址：{{videoUrl.address|noCheck}}<br/>公民身份号码：{{videoUrl.id|noCheck}}</td>
								<td v-else-if="preLookInfo.channel_id==8">
									<span v-for="item in videoUrl" v-if="videoUrl.length>0">{{item}}<br/></span>
									<span v-else>未识别到内容</span>
								</td>
								<td v-else-if="preLookInfo.channel_id==6">
									<span>{{runningTitle.license_type}}：{{videoUrl.license_type}}<br/></span>
									<span>{{runningTitle.plate_no}}：{{videoUrl.plate_no}}<br/></span>
									<span>{{runningTitle.vehicle_type}}：{{videoUrl.vehicle_type}}<br/></span>
									<span>{{runningTitle.owner}}：{{videoUrl.owner}}<br/></span>
									<span>{{runningTitle.address}}：{{videoUrl.address}}<br/></span>
									<span>{{runningTitle.use_character}}：{{videoUrl.use_character}}<br/></span>
									<span>{{runningTitle.model}}：{{videoUrl.model}}<br/></span>
									<span>{{runningTitle.vin}}：{{videoUrl.vin}}<br/></span>
									<span>{{runningTitle.engine_no}}：{{videoUrl.engine_no}}<br/></span>
									<span>{{runningTitle.register_date}}：{{videoUrl.register_date}}<br/></span>
									<span>{{runningTitle.issue_date}}：{{videoUrl.issue_date}}<br/></span>
								</td>
								<td v-else-if="preLookInfo.channel_id==7">
									<div v-if="videoUrl.license_type!=='其他'">
										<span>{{driveTitle.license_type}}：{{videoUrl.license_type}}<br/></span>
										<span>{{driveTitle.card_id}}：{{videoUrl.card_id}}<br/></span>
										<span>{{driveTitle.driver}}：{{videoUrl.driver}}<br/></span>
										<span>{{driveTitle.sex}}：{{videoUrl.sex}}<br/></span>
										<span>{{driveTitle.nationality}}：{{videoUrl.nationality}}<br/></span>
										<span>{{driveTitle.address}}：{{videoUrl.address}}<br/></span>
										<span>{{driveTitle.birthday}}：{{videoUrl.birthday}}<br/></span>
										<span>{{driveTitle.issue_date}}：{{videoUrl.issue_date}}<br/></span>
										<span>{{driveTitle.be_class}}：{{videoUrl.be_class}}<br/></span>
										<span>{{driveTitle.valid_start}}：{{videoUrl.valid_start}}至{{videoUrl.valid_end}}<br/></span>
									</div>
									<div v-else>
										<span>证件类型不是驾驶证，请上传驾驶证照片！</span>
									</div>
								</td>
								<td v-else-if="preLookInfo.channel_id==9">
									<div v-if="videoUrl.license_type!=='其他'">
										<span>{{businessTitle.license_type}}：{{videoUrl.license_type}}<br/></span>
										<span>{{businessTitle.business_id}}：{{videoUrl.business_id}}<br/></span>
										<span>{{businessTitle.business_name}}：{{videoUrl.business_name}}<br/></span>
										<span>{{businessTitle.business_type}}：{{videoUrl.business_type}}<br/></span>
										<span>{{businessTitle.address}}：{{videoUrl.address}}<br/></span>
										<span>{{businessTitle.operator}}：{{videoUrl.operator}}<br/></span>
										<span>{{businessTitle.registered_capital}}：{{videoUrl.registered_capital}}<br/></span>
										<span>{{businessTitle.register_date}}：{{videoUrl.register_date}}<br/></span>
										<span>{{businessTitle.business_term}}：{{videoUrl.business_term}}<br/></span>
										<span>{{businessTitle.scope}}：{{videoUrl.scope}}<br/></span>
									</div>
									<div v-else>
										<span>证件类型不是营业执照，请上传营业执照照片！</span>
									</div>
								</td>
								<td v-else-if="preLookInfo.channel_id==10">
									<div v-if="videoUrl.bank_name !==''">
										<span>{{bankTitle.bank_name}}：{{videoUrl.bank_name|noCheck}}<br/></span>
										<span>{{bankTitle.bank_cardno}}：{{videoUrl.bank_cardno|noCheck}}<br/></span>
										<span>{{bankTitle.expiry_date}}：{{videoUrl.expiry_date|noCheck}}<br/></span>
										<span>{{bankTitle.card_type}}：{{videoUrl.card_type|noCheck}}<br/></span>
										<span>{{bankTitle.card_name}}：{{videoUrl.card_name|noCheck}}<br/></span>
									</div>
									<div v-else>
										<span>证件类型不是银行卡，请上传银行卡照片！</span>
									</div>
								</td>
								<td v-else-if="preLookInfo.channel_id==11">
									<div v-if="videoUrl">
										<span v-for="item in videoUrl.handwritten_content">{{item}}<br/></span>
									</div>
									<div v-else>
										<span>未识别到内容！</span>
									</div>
								</td>
								<td v-else-if="preLookInfo.channel_id==12">
									<div v-if="videoUrl.plate_no !==''">
										<span>{{carTitle.plate_no}}：{{videoUrl.plate_no|noCheck}}<br/></span>
									</div>
									<div v-else>
										<span>照片类型不是车牌或没有识别。</span>
									</div>
								</td>
								<td v-else-if="preLookInfo.channel_id==13">
									<div v-if="videoUrl.business_name !==''">
										<span>{{cardTitle.business_name}}：{{videoUrl.business_name|noCheck}}<br/></span>
										<span>{{cardTitle.position}}：{{videoUrl.position|noCheck}}<br/></span>
										<span>{{cardTitle.company}}：{{videoUrl.company|noCheck}}<br/></span>
										<span>{{cardTitle.address}}：{{videoUrl.address|noCheck}}<br/></span>
										<span>{{cardTitle.email}}：{{videoUrl.email|noCheck}}<br/></span>
										<span>{{cardTitle.phone}}：{{videoUrl.phone|noCheck}}<br/></span>
										<span>{{cardTitle.telephone}}：{{videoUrl.telephone|noCheck}}<br/></span>
										<span>{{cardTitle.qq}}：{{videoUrl.qq|noCheck}}<br/></span>
										<span>{{cardTitle.webchat}}：{{videoUrl.webchat|noCheck}}<br/></span>
									</div>
									<div v-else>
										<span>证件类型不是银行卡，请上传银行卡照片！</span>
									</div>
								</td>
								<td v-else-if="preLookInfo.file_type==3">{{videoUrl.text}}</td>
								<td v-else>无文本内容</td>
							</tr>
						</table>
					</div>

				</el-dialog>
			</div>
			<el-row style="min-width: 800px;">
				<el-col :md={span:20,offset:2} :lg={span:20,offset:2} :xl={span:16,offset:4}>

				</el-col>
			</el-row>
		</div>
		<FooterIndex></FooterIndex>
	</div>

</template>

<script>
    import Navigation from "../components/navigation.vue"
    import FooterIndex from "../components/footerIndex.vue"
    import ImageCheck from '../views/AICheck/imageCheck.vue'
    import AudioCheck from '../views/AICheck/AudioCheck.vue'
    import VideoCheck from '../views/AICheck/videoCheck.vue'
    import TextCheck from '../views/AICheck/textCheck.vue'
    import audioSample from '../components/audioSample.vue'
    import audioExample from '../components/audio.vue'
    import {scrollBy,getDate} from '../store/common'
    export default {
        data() {
            return {
                dialogImageUrl: require("../assets/image/sample_image.png"),
                dialogVisible: false,
                jsonDemo:'{"ret":0,"msg":"ok","data":{"tag_list":[{"tag_name":"protest","probability":0.3087675468623638},{"tag_name":"violence","probability":0.017580988407135},{"tag_name":"sign","probability":0.795149803161621},{"tag_name":"photo","probability":0.0147841582074761},{"tag_name":"fire","probability":0.0184208210557699},{"tag_name":"police","probability":0.0153429144993424},{"tag_name":"children","probability":0.00931504089385271},{"tag_name":"group_20","probability":0.683478415012359},{"tag_name":"group_100","probability":0.207240253686904},{"tag_name":"flag","probability":0.122076518833637},{"tag_name":"night","probability":0.212110564112663},{"tag_name":"shouting","probability":0.0291868895292282}]}}',
                buttonWord:"开始检测",
                showPercent:"概率：1.75%",
                isForce:false,
                imageRight:false,
                imageIsBig:false,
                beginDate:'',
                endDate:'',
                currentPager:1,
                prevText:'上一页',
                nextText:'下一页',
                checkType:1,
                stopVideo:false,
                stopAudio:false,
                centerDialogVisible: false,
                imageFile:'',
                isUploading:false,
                currentValue:null,
                fileUrl:'',
                runningTitle:{license_type:'证件类型',plate_no:'号牌号码',vehicle_type:'车辆类型',owner:'所有人',address:'住址',use_character:'使用性质',model:'品牌型号',vin:'车辆识别代号',engine_no:'发动机号码',register_date:'注册日期',issue_date:'发证日期'},
                driveTitle:{license_type:'证件类型',card_id:'证号',driver:'姓名',sex:'性别',nationality:'国籍',address:'住址',birthday:'出生日期',issue_date:'初次领证日期',be_class:'准驾车型',valid_start:'有效期限'},
                businessTitle:{license_type:'证件类型',business_id:'统一社会信用代码',business_name:'名称',business_type:'类型',address:'住所',operator:'法定代表人',registered_capital:'注册资本',register_date:'成立日期',business_term:'营业期限',scope:'经营范围'},
                bankTitle:{bank_name:'银行信息',bank_cardno:'卡号',expiry_date:'有效期',card_type:'银行卡类型',card_name:'姓名'},
                cardTitle:{business_name:'姓名',position:'职位',company:'公司',address:'地址',email:'邮箱',phone:'手机',telephone:'电话',qq:'QQ',webchat:'微信'},
                carTitle:{plate_no:'车牌'},
                count:0,
                historyInfo:[],
                preLookInfo:{},
                videoUrl:{},
                markerInfo:[],
                player:null,
                fileSort:2,
                firstVideo:true,
                srcList:[],
                audioWord:'',
                pickerOptions: {
                    disabledDate(time) {
                        return time.getTime() > Date.now();
                    }
                },
                pickerOptionsStart: {
                    /*disabledDate(time) {
                        return time.getTime() < new Date(new Date().toLocaleDateString()).getTime();
                    }*/
                    disabledDate(time) {
                        return time.getTime() > Date.now();
                    }
                },
                pickerOptionsEnd: {
                    disabledDate(time) {
                        return time.getTime() > Date.now();
                    }
                },
                searchContent:''
            };
        },
        components:{
            Navigation,
            FooterIndex,
            ImageCheck,
            AudioCheck,
            VideoCheck,
            TextCheck,
            audioSample,
            audioExample

        },
        watch:{

        },
        activated(){
//            this.getHistory(1);
        },
        mounted:function () {
            this.getHistory();
        },
        methods: {
            showInputValue(){
                this.$message.error('该功能尚未开通！')
            },
            changeStart() {
                this.pickerOptionsStart = Object.assign({}, this.pickerOptionsStart, {
                    // 可通过箭头函数的方式访问到this
                    disabledDate: (time) => {
                        //这样写有bug
                        //   return  this.outTime < time.getTime() < new Date(new Date().toLocaleDateString()).getTime()
                        var times = '';
                        //百度没搜到试了好久才试出来
                        times = this.endDate < time.getTime()||time.getTime() > Date.now();
                        //打印了几百条
                        return times
                    }
                })

            },
            //开始时间 控制结束时间
            changeEnd() {

                this.pickerOptionsEnd = Object.assign({}, this.pickerOptionsEnd, {
                    disabledDate: (time) => {
                        return time.getTime() < this.beginDate ||time.getTime() > Date.now();
                    }
                })
            },
            urlCheck(){
                if(this.fileUrl ==''){
                    this.$message.error('请输入要检测的文件地址')
                    return
                }
                let url = this.fileUrl.substring(this.fileUrl.length-5);
                if(url.indexOf("png") !=-1|url.indexOf("jpg") !=-1|url.indexOf("jpeg") !=-1){
                    this.checkType = 1;
                    this.stopVideo = true;
                    this.stopAudio = true;
                    this.submitImageCallback(null,null,this.fileUrl);
                    this.isUploading = true;
                }else if(url.indexOf("wav") !=-1){
                    this.checkType = 3;
                    this.stopVideo = true;
                    this.stopAudio = false;
                    this.submitAudioCallback(null,null,this.fileUrl);
                    this.isUploading = true;
                }else if(url.indexOf("mp4") !=-1){
                    this.checkType = 2;
                    this.stopVideo = false;
                    this.stopAudio = true;
                    this.submitVideoCallback(null,null,this.fileUrl);
                    this.isUploading = true;

                }else if(url.indexOf("text") !=-1){
                    this.checkType = 4;
                    this.stopVideo = true;
                    this.stopAudio = true;
                    this.submitTextCallback(null,null,this.fileUrl);
                    this.isUploading = true;
                }else {
                    this.$showMessageShort('您选择的文件格式错误！');
                }
            },
            searchHistory(){
                if(!this.beginDate&&!this.endDate&&!this.searchContent){
                    this.$showMessageShort('请选择/输入要查询的条件！');
                    return;
                }
                this.getHistory('',this.beginDate,this.endDate,this.searchContent);
            },
            submitImageCallback(e,file,url){
                this.$refs.imageCheck.submitImage(e,file,url);
            },
            submitAudioCallback(e,file,url){
                this.$refs.audioCheck.submitAudio(e,file,url);
            },
            submitVideoCallback(e,file,url){
                this.$refs.videoCheck.submitVideo(e,file,url);
            },
            submitTextCallback(e,file,url){
                this.$refs.textCheck.submitText(e,file,url);
            },
            changeImage(e){
                this.imageIsBig = false;
                this.imageRight = false;
                const file = e.target.files[0];
                const reader = new FileReader();
                const that = this;
                reader.readAsDataURL(file);
                reader.onload  = (e)=> {
                    that.dialogImageUrl = e.target.result;
                    const fileType = file.type;
                    if(fileType.substr(0, 5) === "image"){
                        if(file.size>20971520){
                            this.$showMessageShort('请选择小于20M的图片！');
                            return;
                        }
                        that.checkType = 1;
                        that.stopVideo = true;
                        that.stopAudio = true;
                        this.submitImageCallback(e,file,e.target.result);
                        this.isUploading = true;
                    }else if(fileType.substr(0, 5) === "audio"){
                        if(fileType.indexOf("wav") ==-1){
                            this.$showMessageShort('请选择wav格式的视频！');
                            return;
                        }
                        if(file.size>20971520){
                            this.$showMessageShort('请选择小于20M的音频文件！');
                            return;
                        }
                        that.checkType = 3;
                        that.stopVideo = true;
                        that.stopAudio = false;
                        this.submitAudioCallback(e,file);
                        this.isUploading = true;
                    }else if(fileType.substr(0, 5) === "video"){

                        if(fileType.indexOf("mp4") ==-1){
                            this.$showMessageShort('请选择mp4格式的视频！');
                            return;
                        }
                        if(file.size>52428800){
                            this.$showMessageShort('请选择小于50M的视频！');
                            return;
                        }
                        that.checkType = 2;
                        that.stopVideo = false;
                        that.stopAudio = true;
                        this.submitVideoCallback(e,file);
                        this.isUploading = true;
                        this.toPractice();
                    }else if(fileType.substr(0, 4) === "text"){
                        if(file.size>1048576*10){
                            this.$showMessageShort('请选择小于10M的文件！');
                            return;
                        }
                        that.checkType = 4;
                        that.stopVideo = true;
                        that.stopAudio = true;
                        this.submitTextCallback(e,file);
                        this.isUploading = true;
                    }else{
                        this.$showMessageShort('您选择的文件格式错误！');
                    }
                    document.getElementById("datafile").value=null;

                };
            },
            changeUploadState(isUploading){
                this.isUploading = isUploading;
            },
            toPractice(){
                scrollBy(document.getElementById('practice_title').offsetTop-100);
//                window.scrollBy(0,document.getElementById('practice_title').offsetTop-100)
            },
            getRespectInfo(id){
                let params = `format=json&id=${id}`
                $.ajax({
                    url: this.api+"/api/v1/history/get_historyrecord/",
                    type: "get",
                    data: params,
                    cache: false,
                    contentType: false,
                    processData: false,
                    success:(response)=>{
                        this.videoUrl = response.results.inspection_result;
                        console.log(this.videoUrl);
                        if(this.preLookInfo.file_type ==2){
                            this.markerInfo = [];
                            this.videoUrl.video_evidence_information.forEach(item=>{
                                if( parseFloat(item.porn_sensitivity_level)>parseFloat(item.violence_sensitivity_level)){//色情系数比较高
                                    if(parseFloat(item.porn_sensitivity_level)>80){
                                        this.markerInfo.push({
                                            time:item.sensitivity_time-this.videoUrl.interval/2,
                                            text:"违规",
                                            class:"red_style"
                                        });
                                    }else {
                                        this.markerInfo.push({
                                            time:item.sensitivity_time-this.videoUrl.interval/2,
                                            text:"疑似违规",
                                            class:"orange_style"
                                        });
                                    }

                                }else {
                                    if(parseFloat(item.violence_sensitivity_level)>80){
                                        this.markerInfo.push({
                                            time:parseFloat(item.sensitivity_time)-this.videoUrl.interval/2,
                                            text:"违规",
                                            class:"red_style"
                                        });
                                    }else {
                                        this.markerInfo.push({
                                            time:parseFloat(item.sensitivity_time)-this.videoUrl.interval/2,
                                            text:"疑似违规",
                                            class:"orange_style"
                                        });
                                    }
                                }
                            });
                            if(this.firstVideo){
                                setTimeout(()=>{
                                    this.initVideo(this.markerInfo);
                                    this.firstVideo = false;
                                },2000)
                            }else {
                                setTimeout(()=>{
                                    this.resetMarker(this.markerInfo)
                                },1000)

                            }

                        }
                        if(this.preLookInfo.file_type ==3){
                            this.audioWord =this.videoUrl.text.app_text;
                        }
                    },
                    error:err=>{
                        console.log(err);
                    }
                });
            },
            getHistory(pager,start,end,name){
				let params;
                // let isPro = process.env=== "prod" ? true : false; //正式环境使用
                let isPro =false;  //开发使用
				let channel_id=isPro? '&channel_id=1':'';
                if(pager){
                    start =start?'&begin_time='+getDate(start)+' 00:00:01':'';
                    end =end?'&end_time='+getDate(end)+' 23:59:59':'';
                    name =name?'&file_name='+name:'';
                    params =  'system_id=1'+`&page=${pager}`+start+end+name+channel_id;
                }else if(start|end|name!='') {
                    start =start?'&begin_time='+getDate(start)+' 00:00:01':'';
                    end =end?'&end_time='+getDate(end)+' 23:59:59':'';
                    name =name?'&file_name='+name:'';
                    params ='system_id=1'+start+end+name+channel_id;
                }else {
                    params =  'system_id=1'
                }
                var formData = new FormData();
                formData.append('system_id','pc');
                formData.append('file_type','99');
                $.ajax({
                    url: this.api+"/api/v1/history/get_historyrecord/",
                    type: "get",
                    data: params,
                    cache: false,
                    contentType: false,
                    processData: false,
                    success:(response)=>{
                        this.count = response.count;
                        this.historyInfo = response.results;
                        this.currentPager = pager?pager:1;
                    },
                    error:err=>{
                        console.log(err);
                    }
                });
            },
            handleCurrentChange(val) {
                this.getHistory(val,this.beginDate,this.endDate,this.searchContent);
            },
            closeDialog(){
                this.preLookInfo={};
            },
            preLook(item){
                console.log(item);
                if(item.porn_percent>50){
                    this.$showMessageShort('该记录涉黄违规，不能预览！');
                    return;
                }
                this.srcList = [];
                this.preLookInfo = item;
                this.getRespectInfo(item.id);
                this.fileSort =item.file_type;
                let imgId ;
                if(item.draw_url){
                    this.srcList.push(item.draw_url);
                }else {
                    this.srcList.push(item.file_url);
                }

                this.centerDialogVisible = true;
            },
            initVideo(item){
                this.player = videojs('Video');
                var player = this.player;
                player.markers({
                    markerStyle:{  //标记样式
                        'width':'5px',
                        'border-radius': '5px',
                        /* 'background-color': 'red'*/
                    },
                    markerTip:{  //悬停标记提示对象
                        display:true,  //是否显示markerTips
                        /*
						  用于动态构建标记提示文本的回调函数。
						  只需返回一个字符串，参数标记是传递给插件的标记对象
						 */
                        text: function(marker) {
                            return  marker.text;
                        }
                    },
                    breakOverlay:{  //每个标记的中断覆盖选项
                        display: true,  //是否显示叠加层
                        displayTime: 3,
                        style:{  //为叠加层添加css样式
                            color:"red"
                        },
                        text: function(marker) {  //回调函数动态构建叠加文本
                            return  marker.text;
                        }
                    },
                    onMarkerReached:function(marker, index){  //只要播放到标记的时间间隔，就会出发此回调函数

                    },
                    onMarkerClick:function(marker,index){  //单击标记时会触发此回调函数，
                        /*
						  单击标记的默认行为是在视频中寻找该点，
						  但是，如果onMarkerClick返回false，则将阻止默认行为
						*/
                    },
                    markers:item,
                });
            },
            resetMarker(marker){
                /*this.player = videojs('Video');
                var player = this.player;*/
                this.player.markers.reset(marker)
            },
        }
    }

</script>

<style scoped>
	.current_width_style{width: 1200px;margin: 0 auto;}
	.banner_outer{position: relative;vertical-align: middle;text-align: center;}
	.describe_outer_banner{font-size: 16px;color: white;;display: inline-block;vertical-align: middle;}
	.tech-banner-box{display: inline-block;vertical-align: middle;height: 100%;width: 0;}
	.describe_outer_banner img{width: 40px;height: 40px;margin-right: 40px;}
	.describe_outer_banner img:nth-of-type(4){margin-right: 0;}
	.describe_outer_banner p{}
	.describe_outer_banner p:nth-of-type(1){font-size: 36px;height: 60px;line-height: 60px;margin-bottom: 15px;}
	.describe_outer_banner p:nth-of-type(2){height: 130px;text-align: center;overflow: hidden;width: 580px;line-height: 30px;}

	.top_contain .banner_outer{background-image: url('../assets/image/image_banner.png');min-width: 1300px;}
	.functional_experience{margin: 50px 0;}
	.functional_experience .title{text-align: center;color: #333333;margin: 40px 0;font-size: 36px;}

	.top_suggest{color: #999999;font-size: 14px;line-height: 40px;height: 30px;margin-bottom: 25px;}
	.init_url_style{flex: 1;height: 35px;line-height: 35px;border: 1px solid #E2ECFC;font-size: 15px;padding-left: 10px;background-color: #FAFCFE;width: 685px;}
	.init_url_style:hover{border: 1px solid #C0C4CC;}
	.init_url_style:focus{border: 1px solid #409EFF;}
	.check_style{display:inline-block;height: 33px;line-height: 33px;font-size: 16px;color: #316DFF;border: 2px solid #316DFF;width: 100px;text-align: center;cursor:pointer;}
	.check_style_hidden{display:inline-block;height: 33px;line-height: 33px;font-size: 16px;color: #666666;border: 2px solid #f5f5f5;width: 100px;text-align: center;cursor:pointer;background-color: #f5f5f5}
	.check_style:hover{background-color: #316DFF;color: white;}
	.local_upload{height: 33px;line-height: 33px;font-size: 16px;}
	.local_upload:before{content: "或";margin: 0 25px;}
	.inputfile{z-index: -11111;width: 0px;height:1px;opacity: 0;position: absolute;}
	.local_upload .btn_upload{display:inline-block;height: 33px;line-height: 33px;font-size: 16px;background-color: #316DFF;color:white;border: 2px solid #316DFF;width: 100px;text-align: center;cursor: pointer;}
	.local_upload .btn_uploading{display:inline-block;height: 33px;line-height: 33px;font-size: 16px;background-color: white;color:#666666;border: 2px solid #e1e3e7;width: 100px;text-align: center;cursor: pointer;}
	.local_upload .btn_upload:hover{background-color: #6087F7;color: white;border: 2px solid #6087F7;}
	.show_input_outer{display: flex;}


	.show_search_con{height: 34px;line-height: 34px;padding-left: 20px;letter-spacing:0;font-size:0px;margin-bottom: 10px;}
	.show_search_con >span:first-child{color: #000000;font-size: 14px;padding: 10px;}
	.show_search_con >span:nth-of-type(2){color: #000000;font-size: 14px;padding: 10px;}
	.search_input{display:inline-block;height: 32px;width:315px;line-height: 32px;border: 1px solid #E2ECFC;font-size: 15px;padding-left: 10px;margin-left: 20px;}
	.search_input:hover{border: 1px solid #C0C4CC;}
	.search_input:focus{border: 1px solid #409EFF;}
	.search_btn{display:inline-block;height: 32px;line-height: 32px;font-size: 14px;background-color: #316DFF;color:white;border: 1px solid #316DFF;width: 80px;text-align: center;cursor: pointer;}
	.search_btn:hover{background-color: #6087F7;color: white;border: 1px solid #6087F7;}

	.show_history_title{height: 50px;line-height: 50px;color: #333333;font-size: 16px;background-color: #ebeff8;width: 1200px;display: flex;font-weight: 100;border: 1px solid #ebeff8;}
	.show_history_title th{flex: 0.8;text-align: center;padding-left: 10px;font-weight: 100;}
	.show_history_title th:last-child{flex: 0.5;text-align: center}
	.show_history_title th:first-child{padding-left: 10px;flex: 3;}
	.show_history_title th:nth-of-type(3){flex: 2.5;}
	.show_history_title th:nth-of-type(6){flex: 2;}
	.show_history_title th:nth-of-type(5){flex: 2.5;}

	.show_history_con{height: 50px;line-height: 50px;color: #333333;font-size: 14px;background-color: #fff;width: 1200px;display: flex;border: 1px solid #e3e8f3;border-top: none;}
	.show_history_center{height: 50px;line-height: 50px;color: #333333;font-size: 14px;background-color: #fff;width: 1200px;border: 1px solid #e3e8f3;border-top: none;text-align: center}
	.show_history_con td{flex: 0.8;padding-left: 10px;text-align: center;}
	.show_history_con td:last-child{flex: 0.5;text-align: center;color: #85afff;cursor: pointer;}
	.show_history_con td:first-child{padding-left: 10px;flex: 3;text-align: left;}
	.show_history_con td:nth-of-type(3){flex: 2.5;}
	.show_history_con td:nth-of-type(6){flex: 2;}
	.show_history_con td:nth-of-type(5){flex: 2.5;}
	.orange_color{color: #ffac09;}
	.red_color{color: #ff524a;}
	.green_color{color: #54cd62;}

	/*预览begin*/
	.show_video_outer{height: 440px;}
	.image_con{height:440px;position: relative;overflow: hidden;text-align: center;vertical-align: middle;}
	.image_con img{max-height: 440px;max-width:100%;object-fit: cover;vertical-align: middle;}
	.image_con span{display: inline-block;vertical-align: middle;height: 100%;width: 0;}
	.image_con #Video{width: 100%;height: 400px;}
	.image_con audio{width: 420px;margin-top: 150px;}
	.image_con p{font-size: 16px;margin-top: 15px;padding-right: 100px;}


	.pre_result_con{padding: 2px 30px 40px;}
	.pre_result_con td{border: 1px solid #e3e8f3;height: 46px;}
	.pre_item{width: 100%;}
	.pre_item td:last-child{padding: 20px;text-align: justify}
	.pre_item td:first-child{width: 140px;text-align: center;background-color: #eaf1f6;}
	.pre_item_common{height: 46px;line-height: 46px;}
	.pre_result_item{height: 175px;line-height: 175px;}
	.pre_word_item {height: 150px;line-height: 150px;}
	.result_outer{display: flex;color: #000000;height: 25px;line-height: 25px;width: 35%;}
	.text_result_outer{margin: 25px 20px 25px 20px;width: 25%;}
	.result_outer_notop{margin-top: 0;}
	.result_outer div:first-child{margin-top: 10px;}
	.type_outer>div:nth-of-type(2){margin-top: 20px;}
	.result_outer p:nth-of-type(1){font-size: 14px;margin:0 15px 0 0;}
	.result_outer p:nth-of-type(2){font-size: 14px;flex: 3;text-align: center;line-height: 23px;height: 23px;}
	.result_outer p:nth-of-type(3){font-size: 14px;flex: 4;text-align: center}
	.result_outer .green_style_name{background-color: #54cd62;border: 1px solid #54cd62;color: #fff}
	.result_outer .green_style_number{border: 1px solid #54cd62;color: #54cd62}
	.result_outer .orange_style_name{background-color: #ffac09;border: 1px solid #ffac09;color: #fff}
	.result_outer .orange_style_number{border: 1px solid #ffac09;color: #ffac09}
	.result_outer .red_style_name{background-color: #ff524a;border: 1px solid #ff524a;color: #fff}
	.result_outer .red_style_number{border: 1px solid #ff524a;color: #ff524a}
	.left_50{margin-left: 50px;}
	#web_text{max-height: 800px;overflow-y: scroll;}
	.disAllow{color: #999999!important;cursor:not-allowed!important;}

	.show_pagination{text-align: center;margin-top: 30px;}
</style>