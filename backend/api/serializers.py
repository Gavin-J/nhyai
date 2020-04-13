from django.contrib.auth.models import User, Group
from rest_framework import serializers
from .models import FileUpload, FileImageTerrorismUpload, FileVisionPornUpload
from .models import VideoFileUpload, AudioFileUpload, AudioFileInspection, ImageFileUpload, WordRecognitionInspection
from .models import WordRecognition, OcrGeneral, OcrIDCard, OcrDrivinglicense, OcrVehiclelicense, OcrBusinesslicense, OcrBankcard, OcrVehicleplate,OcrHandWritten,HistoryRecord, OcrBusinessCard


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')


class FileUploadSerializer(serializers.HyperlinkedModelSerializer):

    result = serializers.JSONField(True)

    class Meta:
        model = FileUpload
        fields = ('datafile', 'result')

    def clean_json(self, obj):
        return obj.result


class WordRecognitionSerializer(serializers.HyperlinkedModelSerializer):

    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)

    class Meta:
        model = WordRecognition
        fields = ('text', 'system_id', 'channel_id',
                  'user_id', 'ret', 'msg', 'data')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data


class WordRecognitionInspectionSerializer(serializers.HyperlinkedModelSerializer):

    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)

    class Meta:
        model = WordRecognitionInspection
        fields = ('text', 'system_id', 'channel_id',
                  'user_id', 'text_url', 'ret', 'msg', 'data')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data


class OcrGeneralSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    box = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = OcrGeneral
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data', 'box', 'draw_url')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.box, obj.draw_url


class OcrIDCardSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    box = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = OcrIDCard
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data', 'box', 'draw_url')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.box, obj.draw_url


class FileImageTerrorismUploadSerializer(serializers.HyperlinkedModelSerializer):
    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)

    class Meta:
        model = FileImageTerrorismUpload
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data


class FileVisionPornUploadSerializer(serializers.HyperlinkedModelSerializer):
    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)

    class Meta:
        model = FileVisionPornUpload
        #fields = ('datafile', 'result')
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data


class VideoFileUploadSerializer(serializers.HyperlinkedModelSerializer):

    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)

    class Meta:
        model = VideoFileUpload
        fields = ('id', 'video', 'screenshot','video_url', 'system_id',
                  'channel_id', 'user_id', 'sync',
                  'orientation', 'data', 'ret', 'msg',
                  'is_task', 'serial_number')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data


class AudioFileUploadSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)

    class Meta:
        model = AudioFileUpload
        fields = ('speech', 'speech_url', 'system_id',
                  'channel_id', 'user_id', 'data', 'ret', 'msg')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data


class AudioFileInspectionSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)

    class Meta:
        model = AudioFileInspection
        fields = ('speech', 'speech_url', 'system_id',
                  'channel_id', 'user_id', 'data', 'ret', 'msg')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data


class ImageFileUploadSerializer(serializers.HyperlinkedModelSerializer):
    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)

    class Meta:
        model = ImageFileUpload
        #fields = ('datafile', 'result')
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data


class OcrDrivinglicenseSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    box = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = OcrDrivinglicense
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data', 'box', 'draw_url')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.box, obj.draw_url


class OcrVehiclelicenseSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    box = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = OcrVehiclelicense
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data', 'box', 'draw_url')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.box, obj.draw_url


class OcrBusinesslicenseSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    box = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = OcrBusinesslicense
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data', 'box', 'draw_url')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.box, obj.draw_url


class OcrBankcardSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    box = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = OcrHandWritten
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data', 'box', 'draw_url')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.box, obj.draw_url


class OcrHandWrittenSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    box = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = OcrVehicleplate
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data', 'box', 'draw_url')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.box, obj.draw_url


class OcrVehicleplateSerializer(serializers.HyperlinkedModelSerializer):

    #result = serializers.JSONField(True)
    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    box = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = OcrVehicleplate
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data', 'box', 'draw_url')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.box, obj.draw_url


class OcrBusinessCardSerializer(serializers.HyperlinkedModelSerializer):

    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    box = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = OcrBusinessCard
        fields = ('image', 'image_url', 'system_id',
                  'channel_id', 'user_id', 'ret', 'msg', 'data', 'box', 'draw_url')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.box, obj.draw_url


class HistoryRecordSerializer(serializers.HyperlinkedModelSerializer):

    ret = serializers.JSONField(True)
    msg = serializers.JSONField(True)
    data = serializers.JSONField(True)
    file_id = serializers.JSONField(True)
    file_name = serializers.JSONField(True)
    file_url = serializers.JSONField(True)
    file_type = serializers.JSONField(True)
    inspection_result = serializers.JSONField(True)
    max_sensitivity_type = serializers.JSONField(True)
    max_sensitivity_level = serializers.JSONField(True)
    max_sensitivity_percent = serializers.JSONField(True)
    violence_percent = serializers.JSONField(True)
    violence_sensitivity_level = serializers.JSONField(True)
    porn_percent = serializers.JSONField(True)
    porn_sensitivity_level = serializers.JSONField(True)
    politics_percent = serializers.JSONField(True)
    politics_sensitivity_level = serializers.JSONField(True)
    public_percent = serializers.JSONField(True)
    public_character_level = serializers.JSONField(True)
    content = serializers.JSONField(True)
    web_text = serializers.JSONField(True)
    app_text = serializers.JSONField(True)
    upload_time = serializers.JSONField(True)
    process_status = serializers.JSONField(True)
    system_id = serializers.JSONField(True)
    channel_id = serializers.JSONField(True)
    user_id = serializers.JSONField(True)
    screenshot_url = serializers.JSONField(True)
    duration = serializers.JSONField(True)
    serial_number = serializers.JSONField(True)
    draw_url = serializers.JSONField(True)

    class Meta:
        model = HistoryRecord
        fields = ('id', 'file_id', 'file_name', 'file_url',
                  'file_type', 'inspection_result', 'max_sensitivity_type',
                  'max_sensitivity_level', 'max_sensitivity_percent','violence_percent', 'violence_sensitivity_level',
                  'porn_percent', 'porn_sensitivity_level', 'politics_percent',
                  'politics_sensitivity_level', 'public_percent', 'public_character_level',
                  'content', 'web_text', 'app_text', 'upload_time', 'process_status',
                  'system_id', 'channel_id', 'user_id', 'screenshot_url', 'duration',
                  'serial_number', 'draw_url','ret', 'msg', 'data')

    def clean_json(self, obj):
        return obj.ret, obj.msg, obj.data, obj.file_id, obj.file_name, obj.file_url,
        obj.file_type, obj.inspection_result, obj.max_sensitivity_type,
        obj.max_sensitivity_level, max_sensitivity_percent,obj.violence_percent, obj.violence_sensitivity_level,
        obj.porn_percent, obj.porn_sensitivity_level, obj.politics_percent,
        obj.politics_sensitivity_level, obj.public_percent, obj.public_character_level,
        obj.content, obj.web_text, obj.app_text, obj.upload_time, obj.process_status,
        obj.system_id, obj.channel_id, obj.user_id, obj.screenshot_url, obj.duration,
        obj.serial_number, obj.draw_url
