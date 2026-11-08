# راهنمای اجرای ریویژن

**تمام کارهای ریویژن از همین پوشهٔ `revision` دنبال می‌شود.**
نوت‌بوک‌ها فایل واقعی داخل پوشه‌ها هستند؛ هر فایل `.ipynb` را مستقیم در Kaggle
Import کن. فایل ZIP فقط برای انتقال مجموعه است و برای اجرای این فایل‌ها لازم نیست.

اگر CodeBERT یا C4 خطای `_commit_hash` داده، نوت‌بوک به‌روز را در نشست تازه اجرا کن؛
[راهنمای رفع خطای Hugging Face](00_plan/HF_SETUP_RECOVERY.md) تنظیم نسخه‌ها را توضیح می‌دهد.
برای این اصلاح، ران‌های موفق SPECTRA و baselineهای گرافی نیازی به تکرار ندارند.

## الان چه کاری انجام بدهم؟

ران **CodeNet / SPECTRA full / seed 42** از روی ZIP و checkpoint تأیید شد:
۴۳ epoch کامل، بهترین epoch برابر ۳۳، دقت validation برابر ۷۹٫۹۱٪ و F1 برابر
۸۰٫۶۷٪. دوباره اجرا نکن. خروجی کامل آن، خصوصاً `progress.pt` و `status.json` را
نگه دار. `FINAL_TEST=False` درست است. DeepSim فعلاً با دقت ۸۱٫۳۳٪ جلوتر است؛
این مقایسه فقط validation یک seed است و برتری آماری یا نتیجهٔ test را ثابت نمی‌کند.

در بررسی خروجی‌های محلی در تاریخ ۲۰۲۶-۱۱-۰۸، پنج baseline روی CodeNet شامل
ASTNN، RtvNN، DeepSim، FA-AST-GGNN و FA-AST-GMN کامل و صحت checkpoint آن‌ها
تأیید شد. این پنج ران را دوباره اجرا نکن. CDLH بعد از ۸ epoch کامل و بخشی از
epoch ۹ به محدودیت زمانی نشست رسیده و باید از `progress.pt` ادامه پیدا کند؛
از ابتدا شروع نکن. راهنمای دقیق ادامه و جدول امتیازهای validation در
`C:\PyProjects\spectrals\outputs\revesion\01\RESULTS_REVIEW.md` است.

برای سه ران اینترنت‌دار CodeNet یعنی CodeBERT، C4-adapted و Qwen3-LoRA
گزارش کردی که خروجی ذخیره‌شده داری. آن‌ها را دوباره شروع نکن؛ با
[نوت‌بوک جمع‌آوری گزارش](05_analysis/validation_screening.ipynb) فقط
`revision_small_reports.zip` را دریافت کن. نیازی به دانلود checkpointهای چند
گیگابایتی برای بررسی اولیه نیست. خروجی اصلی را در Kaggle نگه دار.

از ۲۹ ران مرحلهٔ اول، وضعیت ۱۹ ران دیگر هنوز به ما گزارش نشده است:

| مجموعه | فایل‌های قابل اجرا الان | تعداد |
| --- | --- | ---: |
| [AtCoder](01_baselines_seed42/atcoder) | full و هر ۹ baseline | ۱۰ |
| [BigCloneBench](01_baselines_seed42/bigclonebench) | full و هر ۸ baseline | ۹ |
| جمع | وضعیت گزارش‌نشده؛ ادامهٔ CDLH جداست | ۱۹ |

فهرست تک‌تک فایل‌ها در [برنامهٔ اجرای ۲۴تایی](queues/LAUNCH_24.md) و
[نسخهٔ CSV](queues/launch_24.csv) است. این فهرست ران‌های کامل و CDLH ناقص را برای
شروع دوباره پیشنهاد نمی‌کند؛ ادامهٔ CDLH را با checkpoint در یک نشست جدا اجرا کن.
اگر هر کدام را قبلاً شروع کرده‌ای، همان ران را ادامه بده و نسخهٔ تکراری نساز.

## دیتاست قبلی را می‌توانم استفاده کنم؟

اگر قبلاً **همین بستهٔ دقیق** را آپلود کرده‌ای، بله؛ لازم نیست دوباره آپلود شود.
اسم نمایشی دیتاست Kaggle یا صرفاً وجود عبارت V3 کافی نیست؛ فایل‌های داخل بسته
باید با نسخهٔ زیر یکسان باشند. نسخهٔ `v3` نوت‌بوک با `v2` دیتاست دو شمارهٔ مستقل است.

| دیتاست | فایل صحیح | حجم تقریبی | train / valid / test |
| --- | --- | ---: | --- |
| CodeNet | `codenet_revision_common_support.zip` | ۱٫۴۷ GB | 69538 / 14896 / 14906 |
| AtCoder | `atcoder_revision_common_support_v2.zip` | ۱۹۵ MB | 70000 / 15000 / 15000 |
| BigCloneBench | `bigclonebench_revision_common_support_v2.zip` | ۴۵ MB | 70000 / 7129 / 7566 |

مسیر همهٔ ZIPها روی سیستم فعلی:
`C:\PyProjects\spectrals\data\benchmarks`

فایل‌های قدیمی `atcoder.zip`، `bigclonebench.zip` و بسته‌های `common_support`
بدون `v2` برای این جدول استفاده نشوند. AtCoder اصلاح‌شده زوج‌های تکراری حذف‌شده
و نگاشت DDG به AST دارد. BigCloneBench اصلاح‌شده اشتراک کد میان splitها و زوج‌های
تکراری را حذف کرده است. جزئیات و SHA-256 دقیق در
[راهنمای دیتاست](datasets/README_FA.md) و [فهرست نسخه‌ها](datasets/versions.json) است.

## تنظیم هر نوت‌بوک

۱. فقط دیتاست مربوط به همان فایل را Attach کن؛ فایل‌های `metadata.json` و داده‌ها
باید برای loader قابل مشاهده باشند. اگر فقط خود ZIP دیده می‌شود، آن را Extract کن.

۲. GPU را روشن کن. برای CodeBERT، C4-adapted و Qwen3-LoRA اینترنت نیز روشن باشد.
تنظیمات batch و learning rate تولیدشده برای هر روش را حفظ کن.

۳. برای ران تازه، `SEED=42`، `RESUME_CHECKPOINT=None`،
`ALLOW_NUMERICAL_UPGRADE=False` و `FINAL_TEST=False` بماند.
عدد ۱۰۰ سقف epoch است؛ early stopping می‌تواند زودتر تمام کند.

۴. Save Version → Save & Run All. فایل خروجی هر مدل را جدا نگه دار.
اسلات‌های موازی مستقل‌اند؛ دو GPU یک نوت‌بوک در کد فعلی به معنای دو برابر شدن سرعت نیست.

۵. اگر `status="incomplete_session_budget"` شد، همان نوت‌بوک و تنظیمات را با
checkpoint ادامه بده. ران ناقص را کامل ثبت نکن. سرعت هر baseline هنوز باید روی
Kaggle سنجیده شود؛ زمان حدود سه‌ساعتهٔ SPECTRA تضمینی برای بقیه نیست.

## نقشهٔ مراحل و هدف هر کدام

| پوشه | چه زمانی؟ | کدام ایراد را بررسی می‌کند؟ |
| --- | --- | --- |
| [00_plan](00_plan/README.md) | راهنما و مراجعه | برنامهٔ کامل، پروتکل، نگاشت ایرادهای داور و محدودیت‌های باز |
| [01_baselines_seed42](01_baselines_seed42/README.md) | **الان** | مقایسهٔ منصفانه روی دادهٔ اصلاح‌شده، baselineهای جدید، آموزش فراتر از ۴ epoch؛ داورهای ۲، ۳ و ۶ |
| [02_controls_seed42](02_controls_seed42/README.md) | پس از screening | آیا اثر از طیف است یا encoder و lossها؟ کنترل معماری و حذف loss؛ محور ایراد داور ۶ |
| [03_selected_repeats](03_selected_repeats/README.md) | پس از انتخاب روی validation | پایداری نتایج مرکزی بین seedها؛ فایل‌های این پوشه گزینه‌اند و همه نباید اجرا شوند |
| [04_transfer](04_transfer/README.md) | پس از انتخاب مدل‌ها | source-only، بودجهٔ داده و تعداد گام برابر، توضیح نظارت زبان مقصد؛ داورهای ۲ و ۶ |
| [05_analysis](05_analysis/README.md) | collector الان؛ تحلیل‌های دیگر بعداً | جمع‌بندی validation، شواهد کمی طیف، هزینه و عدم‌قطعیت؛ سپس test پس از تثبیت تصمیم‌ها |
| بازنویسی مقاله | بر اساس نتایج | اصلاح ادعاها، novelty، متن، شکل‌ها و پاسخ داورها؛ صرف اجرای نوت‌بوک این بخش را حل نمی‌کند |

کتابخانهٔ فعلی ۱۹۳ نوت‌بوک دارد؛ این عدد تعداد ران‌های لازم نیست.
جزئیات برنامه در [پلن کامل فارسی](00_plan/REVISION_PLAN_FA.md) و
[پلن مرجع](00_plan/REVISION_MASTER_PLAN.md) است. بازتولید دقیق FSD-CLCD/ZC3،
برچسب‌های رسمی clone type و متن قابل ویرایش مقاله همچنان موضوعات جداگانه و باز هستند.

کد، وزن‌ها، دیتاست و checkpoint با این مرتب‌سازی عوض نشده‌اند.
مسیر قدیم و جدید هر فایل در [جدول جابه‌جایی](queues/path_migration.csv) ثبت شده است.
