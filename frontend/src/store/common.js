import Vue from 'vue'
import moment from 'moment'

export default {
	install(Vue, options) {
		//接口地址
		// Vue.prototype.api = "http://www.ischoolhn.com";	//测试环境
        //内网环境
        // Vue.prototype.api = "http://220.174.232.142:9016";	//测试环境

		// Vue.prototype.api = "http://www.ischoolhn.com";	//试运行环境
		Vue.prototype.api = "http://172.31.4.31:8000";	//书径
		// Vue.prototype.api = "http://172.31.20.59:8000";	//卫俊
		// Vue.prototype.api = "https://ai.hn-ssc.com";	//卫俊
		Vue.prototype.imgApi = "https://ai.hn-ssc.com";

		//Vue.prototype.api = "http://www.hn-ssc.com";	//正式环境
        // Vue.prototype.api = "";	//打包环境

		//通用错误处理
		Vue.prototype.getError = function(data) {
			console.log(data);
			if(data.code == 1000) {
				store.commit(types.LOGOUT);
			} else {
				alert(data.msg);
			}
		};
	},
}

//过滤器
Vue.filter('momentDate', function(time, dateType) {
    // 返回处理后的值

    if(time){
        time = time.replace('T'," ");
        let number = time.indexOf('.');
        if(number!=-1){
            time = time.substring(0 ,number);
            console.log('substring',time);
        }
        time = time.replace(/-/g,"/");
    }else {
        return;
    }
    var date = new Date(time);
    if(dateType == null) {
        return moment(date).format('YYYY-MM-DD HH:mm:ss')
    } else {
        return moment(date).format(dateType);
    }
});
//过滤器
Vue.filter('noCheck', function(name) {
    // 返回处理后的值
    if(name){
    	return name;
	}else {
        return '未识别';
	}
});
//过滤器
Vue.filter('formatDate', function(time, dateType) {
	// 返回处理后的值
    if(time){
        time = time.replace(/-/g,"/");
    }else {
    	return;
	}
	var date = new Date(time);
	if(dateType == null) {
		return formatDate(date, 'yyyy-MM-dd');
	} else {
		return formatDate(date, dateType);
	}
});
export function getDate(msg,dateType) {
    var date = new Date(msg);
    if(dateType == null) {
        return formatDate(date, 'yyyy-MM-dd');
    } else {
        return formatDate(date, dateType);
    }
};
export function showMessageShort(msg) {
    this.$message.closeAll();
    this.$message.error({
        showClose: true,
        message: msg,
        type: 'error',
        duration:1500
    });
};
export function canvasBox(boxes,src,image,canvas,callback) {
    var W = image.naturalWidth;
    var H = image.naturalHeight;
    var c = canvas;
    var ctx = c.getContext("2d");
    var img = new Image();
    img.src =src;
    img.onload = ()=> {
        c.height = H;
        c.width = W;
        ctx.strokeStyle = "blue";
        ctx.lineWidth =1;
        ctx.drawImage(img, 0, 0);
        ctx.beginPath();
        //图片加载完之后再画矩形
        //我这里用的是rect方法，这里可以替换成你写的mveto和lineto的方法
        for(var i=0;i<boxes.length;i++){
            var x1 = parseInt(boxes[i][0])-0.5;
            var y1 = parseInt(boxes[i][1])-0.5;
            var x2 = parseInt(boxes[i][2])-0.5;
            var y2 = parseInt(boxes[i][3])-0.5;
            var x3 = parseInt(boxes[i][4])-2;
            var y3 = parseInt(boxes[i][5])-2;
            var x4 = parseInt(boxes[i][6])-0.5;
            var y4 = parseInt(boxes[i][7])-0.5;
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.lineTo(x3, y3);
            ctx.lineTo(x4, y4);
            ctx.lineTo(x1, y1);
            // ctx.moveTo(x1, y1);
            // ctx.lineTo(x2, y2);
            // ctx.lineTo(x3, y3);
            // ctx.lineTo(x4, y4);
            // ctx.lineTo(x1, y1);
        }
        ctx.stroke();
        ctx.closePath();
        callback();
    }
};
export function clearCanvas(canvas){
    var c=document.getElementById("myCanvas");
    var cxt=c.getContext("2d");
    c.height=c.height;
};

export function formatDate(date, fmt) {//2018-03-21 18:08:48
	if(/(y+)/.test(fmt)) {
		fmt = fmt.replace(RegExp.$1, (date.getFullYear() + '').substr(4 - RegExp.$1.length));
	}
	let o = {
		'M+': date.getMonth() + 1,
		'd+': date.getDate(),
		'h+': date.getHours(),
		'm+': date.getMinutes(),
		's+': date.getSeconds()
	};
	for(let k in o) {
		if(new RegExp(`(${k})`).test(fmt)) {
			let str = o[k] + '';
			fmt = fmt.replace(RegExp.$1, (RegExp.$1.length === 1) ? str : padLeftZero(str));
		}
	}
	return fmt;
};

function padLeftZero(str) {
	return('00' + str).substr(str.length);
};
export function addDate() {
    let nowDate = new Date();
    let date = {
        year: nowDate.getFullYear(),
        month: nowDate.getMonth() + 1,
        date: nowDate.getDate(),
    }
    date.month = date.month>9?date.month:0+date.month;
    date.date = date.date>9?date.date:0+date.date;
    console.log(date);
    return date.year + '-' + date.month + '-' + date.date;
}
export function scrollBy(offsetTop){
    // window.scrollBy(0,offsetTop);
    $('html,body').animate({ scrollTop: offsetTop}, 200)
}
//过滤器
Vue.filter('secondToTime', function(second) {
    // 返回处理后的值
    return secondToTime(second);
});
export function secondToTime(second) {
	const hour = second>3600?Math.floor(second/3600)+":":"00:";
	const minute = second>60?Math.floor(second/60)>10?Math.floor(second/60):"0"+Math.floor(second/60)+":":"00:";
	const sec = second%60<=9?"0"+Math.ceil(second%60):Math.ceil(second%60);
	return hour  + minute + sec;
}

//    封装禁止页面滚动方法（该方法兼容PC端和移动端）
export function stopBodyScroll (isFixed,top) {
    var bodyEl = document.body
    if (isFixed) {
        bodyEl.style.position = 'fixed'
        bodyEl.style.top = -top + 'px'
    } else {
        bodyEl.style.position = ''
        bodyEl.style.top = ''
        window.scrollTo(0, top) // 回到原先的top
        window.scrollTo(0, top) // 回到原先的top
    }
}




