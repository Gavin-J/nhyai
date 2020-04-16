// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import {TimePicker,Row,Col,Progress,DatePicker,Pagination,Dialog,Upload,Carousel,CarouselItem,Loading,Image} from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import moment from 'moment'
import App from './App'
import router from './store/router'
import common from './store/common'
import fileUtil from './store/fileUtil'
import axios from 'axios'
import { post, fetch, patch, put} from './store/https'
import {showMessageShort} from './store/common'
import AudioRecorder from 'vue-audio-recorder'
import "babel-polyfill"

//定义全局变量
Vue.prototype.$post = post;
Vue.prototype.$fetch = fetch;
Vue.prototype.$patch = patch;
Vue.prototype.$put = put;
Vue.prototype.$axios = axios;
Vue.prototype.$showMessageShort = showMessageShort;

Vue.use(Col);
Vue.use(Row);
Vue.use(TimePicker);
Vue.use(Progress);
Vue.use(DatePicker);
Vue.use(Pagination);
Vue.use(Dialog);
Vue.use(Upload);
Vue.use(Carousel);
Vue.use(CarouselItem);
Vue.use(Loading);
Vue.use(Image);
Vue.use(common);
Vue.use(fileUtil);
Vue.use(AudioRecorder);
Vue.use(moment);

Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})

router.afterEach((to, from, next) => {
  setTimeout(() => {
    var _hmt = _hmt || [];
    (function () {
      //每次执行前，先移除上次插入的代码
      document.getElementById('baidu_tj') && document.getElementById('baidu_tj').remove();
      var hm = document.createElement("script");
      hm.src = "https://hm.baidu.com/hm.js?11168791855c1b5ac5bcb8c26e91fd1b";
      hm.id = "baidu_tj"
      var s = document.getElementsByTagName("script")[0];
      s.parentNode.insertBefore(hm, s);
    })();
  }, 0);
});
