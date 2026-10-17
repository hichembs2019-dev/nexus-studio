# nexus-studio

وكيل إخراج يحوّل فكرة قصة إلى فيديو حقيقي، باستخدام نموذج **Wan 2.2 TI2V-5B**
مفتوح المصدر، يعمل على خادم GPU حقيقي (RunPod) عبر ComfyUI - بدون أي اشتراك
في واجهة فيديو مدفوعة (Kling/Runway/Veo).

## الحقيقة المثبتة فعلياً (2026-10-17)

كل رقم هنا تم قياسه حقيقياً، لا تقديراً - التفاصيل والدليل الكامل في
`docs/SOURCES.md`.

- توليد حقيقي واحد (5.04 ثانية فيديو، 121 فريم @ 24fps، دقة 1280×704) استغرق
  **202.87 ثانية** على RTX 4090 حقيقي.
- تكلفة الخادم: **$0.89/ساعة** (RunPod Secure Cloud).
- **كل ساعة GPU تعطي تقريباً 1.5 دقيقة فيديو نهائي** بهذه الدقة/عدد الخطوات -
  وهذا سقف التكلفة الخام للتوليد فقط، بدون كتابة السيناريو أو المونتاج.

## البنية

يتبع نفس اصطلاح nexus-commerce: كل تكامل خارجي له Interface + تنفيذ حقيقي +
تنفيذ وهمي (Mock) للاختبارات، ولا شيء "يعمل" بدون دليل حقيقي أولاً.

```
src/
  models.py                          # Scene, Episode, ProducedEpisode
  agents/director.py                  # وكيل الإخراج: فكرة -> مشاهد
  workflows/
    wan_prompt_builder.py             # الرسم البياني الحقيقي المُثبت لـ Wan 2.2
    episode_pipeline.py               # ينفّذ كل مشهد ثم يدمجها
  integrations/
    comfyui_client.py                 # /prompt, /history, /view الحقيقية
    remote_exec_client.py             # تنفيذ أوامر على الخادم بدون SSH (محظور)
    reasoning_client.py               # Groq - تخطيط المشاهد
```

- **CLIP_SECONDS** (`src/models.py`) ثابت حقيقي: كل استدعاء توليد واحد = 5.04
  ثانية فيديو. عدد المشاهد في أي حلقة = مدتها المطلوبة ÷ 5.04، وليس اختياراً
  حراً.
- **SSH محظور** من بيئة التنفيذ (port 22 - مؤكد بالفحص الحقيقي، ليس افتراضاً)،
  فالدمج النهائي للمشاهد (ffmpeg) ينفَّذ على الخادم نفسه عبر Jupyter
  (`remote_exec_client.py`)، لا عبر SSH.

## التشغيل

```bash
pip install -r requirements.txt
cp .env.example .env   # ثم عبّئ القيم الحقيقية
pytest -q              # 29 اختباراً، كلها بـ Mock - لا تكلفة حقيقية
```

لتشغيل حلقة حقيقية:

```python
from src.config import Settings
from src.app import build_app

app = build_app(Settings.from_env())
episode = app.director.plan_episode("Mars Walk", "رائد فضاء يستكشف المريخ لأول مرة.", target_seconds=30)
produced = app.pipeline.produce(episode)
print(produced.stitched_path, produced.total_cost_usd)
```

## الحالة

- ✅ التوليد الحقيقي لمشهد واحد: مُثبت بالكامل (زمن + تكلفة + ملف فيديو حقيقي).
- ✅ وكيل الإخراج (تخطيط المشاهد من فكرة): مبني ومُختبر.
- ✅ خط أنابيب الحلقة الكاملة (توليد كل مشهد + دمجها): مبني ومُختبر بـ Mock.
- ⏳ لم يُشغَّل بعد: حلقة حقيقية كاملة متعددة المشاهد من بداية لنهاية على
  الخادم الحقيقي (الخطوة التالية المنطقية).
- ⏳ النشر التلقائي على المنصات (YouTube/TikTok/X/...): غير مبني بعد.
