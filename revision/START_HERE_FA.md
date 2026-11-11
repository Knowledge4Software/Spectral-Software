# راهنمای اجرای ریویژن

برای پیدا کردن ران‌های فعلی از [صف مستقل و زمان هر نوت‌بوک](queues/RUN_NEXT_INDEPENDENT_BATCH.md)
و [صف baselineهای باقی‌مانده](queues/LAUNCH_24.md) استفاده کن. وضعیت واقعی هر ران
در فایل‌های CSV داخل [results](results/README.md) ثبت شده است؛ نبود خروجی به معنی شروع‌نشدن ران نیست.
اگر ران روی Kaggle در حال اجراست، نسخهٔ دوم شروع نکن.

## وضعیت تأییدشده

| بخش | وضعیت |
| --- | --- |
| شش حالت اصلی CodeNet با seed=42 | کامل؛ نتایج قبلی را استفاده کن |
| همان شش حالت با seedهای 123 و 231 | هر ۱۲ تکرار کامل؛ هستهٔ اصلی مجموعاً ۱۸ نتیجه دارد |
| کنترل‌های seed=42 | ۱۰ مورد از ۲۰ کنترل کامل؛ ۱۰ خروجی loss/weight هنوز ارسال نشده |
| full روی AtCoder و BigCloneBench با seedهای 123 و 231 | چهار خروجی هنوز ارسال نشده |
| صف اولیهٔ ۲۹ baseline/full | ۱۸ کامل، چهار ناقص و هفت مورد بدون خروجی ارسالی |

[گزارش سه‌seed هستهٔ اصلی](results/CORE_REPEATS_PROGRESS.md) و
[گزارش کنترل‌های گراف/توصیفگر](results/ARCHITECTURE_CONTROLS_REVIEW.md) جزئیات را دارند.
ران classification-only/123 پس از ۵۳ epoch، با بهترین epoch برابر ۴۳، دقت 80.58٪
و F1 برابر 80.99٪ کامل شد. ران خطادار قبلی جداگانه حفظ شده و seed اضافی محسوب نمی‌شود.

هر پنج کنترل projected، projected_ast، projected_ddg، energy_only و density_heat کامل‌اند.
نتیجهٔ ضعیف DDG با دقت 56.00٪ را نگه می‌داریم؛ علت آن از بررسی فایل‌ها به‌عنوان باگ دیتاست ثابت نشده است.
نتایج این مرحله validation هستند؛ `FINAL_TEST=False` بماند.

## چه خروجی‌هایی هنوز ارسال نشده؟

- CodeNet، همگی seed=42: `without_spectral`، `without_auc_ranking`،
  `without_hard_negative`، `without_reconstruction`، `without_topology_reconstruction`،
  `without_spectral_variance`، `without_graph`، `without_spectral_and_hard_negative`،
  `auxiliary_half` و `auxiliary_double`.
- AtCoder/full و BigCloneBench/full، هرکدام seedهای 123 و 231.
- baselineهای قدیمی‌تر، همگی seed=42: AtCoder شامل CodeBERT، C4-adapted، CDLH و Qwen؛
  BigCloneBench شامل CodeBERT، CDLH و Qwen.

برای CodeNet/CodeBERT، C4، CDLH و Qwen قبلاً خروجی ناقص ارسال شده؛ خروجی کامل ادامهٔ آن‌ها هنوز نداریم.
طبق [راهنمای ادامهٔ مدل‌های متنی](00_plan/TEXT_RUN_CONTINUATION.md)، checkpoint و token cache
همان روش را وصل کن. مدل‌های قدیمی از [نسخهٔ آرشیوشدهٔ خودشان](archive/input_v1/README.md) ادامه پیدا می‌کنند.

## فایل و دیتاست درست

فایل‌های `.ipynb` لینک‌شده در صف فعلی، نسخه‌های اصلاح‌شدهٔ عددی هستند و مستقیماً در Kaggle Import می‌شوند.
فقط دیتاست مربوط به همان نوت‌بوک را وصل کن؛ همان آپلودهای قبلی کافی‌اند.

| دیتاست | بستهٔ دقیق | train / valid / test |
| --- | --- | --- |
| CodeNet | `codenet_revision_common_support.zip` | 69538 / 14896 / 14906 |
| AtCoder | `atcoder_revision_common_support_v2.zip` | 70000 / 15000 / 15000 |
| BigCloneBench | `bigclonebench_revision_common_support_v2.zip` | 70000 / 7129 / 7566 |

بسته‌ها در `C:\PyProjects\spectrals\data\benchmarks` هستند؛ SHA-256 در
[راهنمای دیتاست](datasets/README_FA.md) و [نسخه‌ها](datasets/versions.json) ثبت شده است.

برای ران تازهٔ صف مستقل: GPU روشن، Internet خاموش، `RESUME_CHECKPOINT=None`،
`ALLOW_NUMERICAL_UPGRADE=False` و `FINAL_TEST=False` بماند. seed مشخص‌شدهٔ هر فایل را تغییر نده.
سقف ۱۰۰ epoch، حداقل ۱۰ و patience برابر ۱۰ حفظ شود. هر نشست ۱۰ ساعت بودجه دارد که
۴۵ دقیقهٔ آن ذخیرهٔ خروجی است؛ رسیدن به سقف زمان به معنی تکمیل آزمایش نیست.
در صورت توقف بودجه، همان ران را با checkpoint خودش ادامه بده.

زمان تخمینی هر کنترل CodeNet روی یک T4 حدود ۲ تا ۸ ساعت، هر تکرار AtCoder حدود ۲ تا ۶ ساعت
و BigCloneBench حدود ۱ تا ۴ ساعت است. این‌ها تخمین‌اند؛ کد فعلی از یک GPU استفاده می‌کند.
مدل‌های متنی ممکن است به چند نشست نیاز داشته باشند. برای فرستادن گزارش کوچک از
[جمع‌کنندهٔ گزارش](05_analysis/validation_screening.ipynb) استفاده کن؛ checkpointهای بزرگ می‌توانند روی Kaggle بمانند.

## مسیرها و نگهداری

| مسیر | کاربرد |
| --- | --- |
| [queues](queues/README.md) | صف‌های اجرا و اثرانگشت نوت‌بوک‌ها |
| [results](results/README.md) | گزارش‌های تأییدشده و وضعیت ران‌ها |
| [00_plan](00_plan/README.md) | پروتکل، پلن و پوشش ایرادهای داور |
| [04_transfer](04_transfer/README.md) | مرحلهٔ بعد از تکمیل baselineها و انتخاب مقایسه‌ها |
| [05_analysis](05_analysis/README.md) | تحلیل، هزینه و عدم قطعیت |
| [نگهداری پروژه](MAINTENANCE.md) | پاک‌سازی امن فایل‌های موقت و حفظ نسخه‌های علمی |

نوت‌بوک‌ها و manifest اصلی برای بازتولید نتایج ثابت نگه داشته می‌شوند.
نسخه‌های تاریخی و نتایج نامطلوب، فایل بی‌استفاده محسوب نمی‌شوند.
اسناد قدیمی شروع ران از راهنمای فعال حذف شده‌اند؛ سابقهٔ آن‌ها در Git محفوظ است.
