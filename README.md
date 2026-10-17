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
  models.py                          # Screenplay, Scene, Episode, ProducedEpisode
  agents/
    writer.py                         # وكيل الكتابة: فكرة -> سيناريو مقروء (يُعدَّل)
    director.py                        # وكيل الإخراج: سيناريو معتمد -> مشاهد بصرية
  workflows/
    wan_prompt_builder.py             # الرسم البياني الحقيقي المُثبت لـ Wan 2.2
    episode_pipeline.py               # ينفّذ كل مشهد ثم يدمجها
  agents/publisher.py                 # وكيل النشر: الحلقة -> رفع حقيقي للمنصات
  bot/
    conversation.py                   # منطق البوت الكامل (مُختبر 100%، بلا مكتبة Telegram)
    runner.py                         # الربط الفعلي مع Telegram (غير مُختبر وحدوياً، يحتاج بوت حقيقي)
  integrations/
    comfyui_client.py                 # /prompt, /history, /view الحقيقية
    remote_exec_client.py             # تنفيذ أوامر على الخادم بدون SSH (محظور)
    reasoning_client.py               # Groq (أساسي) + Gemini (احتياطي) - الكتابة والإخراج
    publishing_client.py              # YouTube (رسم حقيقي من التوثيق الرسمي) - منصات أخرى لم تُبنَ بعد
    telegram_client.py                # إرسال رسائل/فيديوهات Telegram
main.py                               # نقطة تشغيل البوت كخدمة دائمة (python -m src.main)
```

**سير العمل الحقيقي (كما طلبه المالك):** فكرة → **الكاتب** يكتب سيناريو كامل
مقروء بلغة الفكرة نفسها (`WriterAgent.write_screenplay`) → **المالك يقرأ
ويعدّل** نص كل مشهد مباشرة (`ScreenplayScene.text`) → عند الرضا، المالك يستدعي
**المخرج** (`DirectorAgent.plan_episode_from_screenplay`) الذي يحوّل كل بيت من
السيناريو المعتمد إلى وصف بصري بالإنجليزية جاهز للتوليد → `EpisodePipeline`
يولّد وينفّذ. لا تسلسل تلقائي بين الكاتب والمخرج - الموافقة خطوة بشرية صريحة.

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
pytest -q              # 70 اختباراً، كلها بـ Mock - لا تكلفة حقيقية
```

## استعمال البوت (الطريقة الدائمة)

بعد تعبئة `.env` (يحتاج `TELEGRAM_BOT_TOKEN` حقيقياً، احصل عليه من
[@BotFather](https://t.me/BotFather) بنفس طريقة nexus-commerce):

```bash
docker compose up -d --build
```

ثم في محادثة البوت على Telegram:

| الأمر | الوظيفة |
| --- | --- |
| `/new [ثواني] <فكرة>` | يطلب من **الكاتب** سيناريو (افتراضياً 30 ثانية إن لم تُحدَّد) |
| `/edit <رقم المشهد> <نص جديد>` | يعدّل مشهداً واحداً في السيناريو الحالي |
| زر **✅ Approve** | **المخرج** يحوّل السيناريو، ثم يبدأ التوليد الحقيقي (دقائق) |
| زر **📤 Publish to YouTube** | ينشر الحلقة الجاهزة فعلياً على قناة Nexora |
| `/cancel` أو زر **❌/🗑️** | يلغي المسودة الحالية |

كل منطق القرار (`src/bot/conversation.py`) مُختبر بالكامل بدون أي بوت حقيقي؛
الربط الفعلي مع Telegram (`src/bot/runner.py`) يحتاج بوتاً حقيقياً ليُختبر،
بنفس المنطق الذي يتبعه nexus-commerce لنفس السبب.

## التشغيل البرمجي المباشر (بدون بوت)

لتشغيل حلقة حقيقية كاملة (الكاتب → المخرج → التوليد → النشر) من كود Python
مباشرة، دون المرور بالبوت:

```python
from src.config import Settings
from src.app import build_app

app = build_app(Settings.from_env())

screenplay = app.writer.write_screenplay("Mars Walk", "رائد فضاء يستكشف المريخ لأول مرة.", target_seconds=30)
for s in screenplay.scenes:
    print(s.index, s.text)          # اقرأ وعدّل s.text هنا كما تشاء قبل أي توليد

episode = app.director.plan_episode_from_screenplay(screenplay)   # بعد الموافقة فقط
produced = app.pipeline.produce(episode)
print(produced.stitched_path, produced.total_cost_usd)

results = app.publisher.publish_episode(produced, ["youtube"])    # يحتاج YOUTUBE_* حقيقية في .env
for r in results:
    print(r.platform, r.success, r.video_id or r.error)
```

## الحالة

- ✅ التوليد الحقيقي لمشهد واحد ولحلقة من مشهدين مع دمج حقيقي: مُثبت بالكامل
  (زمن + تكلفة + ملف فيديو حقيقي قابل للعب - اختبار 2026-10-17 الحقيقي كلّف
  $0.083 فعلياً).
- ✅ وكيل الكتابة (فكرة -> سيناريو مقروء): مبني ومُختبر، ومُختبر حقيقياً (عربي →
  سيناريو عربي متماسك).
- ✅ وكيل الإخراج (سيناريو معتمد -> مشاهد بصرية): مبني ومُختبر، ومُختبر حقيقياً.
- ✅ خط أنابيب الحلقة الكاملة (توليد كل مشهد + دمجها): مُثبت على الخادم الحقيقي.
- ✅ احتياطي Groq → Gemini عند تجاوز الحصة اليومية: مبني ومُختبر حقيقياً.
- ✅ وكيل النشر (`PublishingAgent`) + **YouTube**: مبني، ومُختبر حقيقياً بالكامل
  على قناة حقيقية فعلياً تحمل اسمها وشعارها الخاص **Nexora** (لا الحساب
  الشخصي للمالك) - رفعان حقيقيان ناجحان (3.37s و2.96s)، مؤكَّدان عبر YouTube
  Studio أنهما يستهدفان Nexora تحديداً. كل خطوة إعداد OAuth الحقيقية موثّقة
  في `docs/SOURCES.md` "Nexora channel setup" (مشروع Google Cloud، شاشة
  الموافقة، الحصول على refresh_token، وكل خطأ حقيقي واجهناه وحله).
- ⏳ TikTok/X/Instagram/Snapchat/Twitch/Kick: لا شيء مبني بعد - التوثيق الرسمي
  لكل منها لم يُقرأ بعد، فلا تخمين لشكل الطلب.
- 🟡 **بوت Telegram الدائم**: مبني بالكامل ومُختبر منطقياً (18 اختباراً على
  `BotConversation` وحده، 70 إجمالاً)، لكن **لم يُشغَّل بعد ببوت حقيقي ولا
  نُشر كخدمة دائمة** - ينتظر `TELEGRAM_BOT_TOKEN` حقيقياً من المالك وقراراً
  بمكان التشغيل (نفس خادم nexus-commerce أم خادم جديد).
