import 'app_localizations.dart';

/// The translations for English (`en`).
class AppLocalizationsEn extends AppLocalizations {
  AppLocalizationsEn([String locale = 'en']) : super(locale);

  @override
  String get appTitle => 'MedAssist';

  @override
  String get loginTitle => 'Sign in';

  @override
  String get loginSubtitle => 'Medicine information you can rely on';

  @override
  String get usernameLabel => 'Username';

  @override
  String get passwordLabel => 'Password';

  @override
  String get emailLabel => 'Email';

  @override
  String get signIn => 'Sign in';

  @override
  String get signUp => 'Sign up';

  @override
  String get signUpTitle => 'Create an account';

  @override
  String get signUpLink => 'Create an account';

  @override
  String get showPassword => 'Show password';

  @override
  String get hidePassword => 'Hide password';

  @override
  String get signInFailed => 'Sign-in failed.';

  @override
  String get signUpFailed => 'Sign-up failed.';

  @override
  String get processFailed => 'Could not analyse the image.';

  @override
  String get createAccount => 'Create an account';

  @override
  String get signOut => 'Sign out';

  @override
  String get signOutConfirmTitle => 'Sign out?';

  @override
  String get signOutConfirmBody => 'You will need to sign in again to continue.';

  @override
  String get cancel => 'Cancel';

  @override
  String get confirm => 'Confirm';

  @override
  String get delete => 'Delete';

  @override
  String get processImageTitle => 'Scan medicine';

  @override
  String get pickImage => 'Pick image';

  @override
  String get processImage => 'Analyse';

  @override
  String get noImagePicked => 'No image selected yet';

  @override
  String get history => 'History';

  @override
  String get search => 'Search';

  @override
  String get resultsTitle => 'Results:';

  @override
  String get noResults => 'No results.';

  @override
  String get processSuccess => 'Done!';

  @override
  String get searchTitle => 'Search medicine';

  @override
  String get searchHint => 'Medicine name';

  @override
  String get searchEmptyQuery => 'Please enter a medicine name.';

  @override
  String get historyTitle => 'Processing history';

  @override
  String get historyEmpty => 'No processing history';

  @override
  String get historyLoadFailed => 'Could not load history.';

  @override
  String get searchResultsLabel => 'Search results:';

  @override
  String get historyFile => 'File:';

  @override
  String get deleteRecordTitle => 'Delete this record?';

  @override
  String get deleteRecordBody => 'This record will be removed from your history.';

  @override
  String get deleted => 'Deleted';

  @override
  String get deleteFailed => 'Could not delete the record';

  @override
  String get retry => 'Retry';

  @override
  String get newsTitle => 'Medicine & health news';

  @override
  String get newsEmpty => 'No news yet.';

  @override
  String get newsLoadFailed => 'Could not load news.';

  @override
  String get newsSampleBanner => 'Showing sample content — a real news source isn\'t enabled yet.';

  @override
  String get newsOpenSource => 'View source';

  @override
  String get newsNoSourcesConfigured => 'The project owner hasn\'t enabled a news source yet. See docs/NEWS.md.';

  @override
  String get offline => 'Offline';

  @override
  String offlineUpdated(String time) {
    return 'Offline · updated $time';
  }

  @override
  String get noInformation => 'No information';

  @override
  String get medicineName => 'Medicine name';

  @override
  String get composition => 'Ingredients';

  @override
  String get uses => 'Uses';

  @override
  String get sideEffects => 'Side effects';

  @override
  String get manufacturer => 'Manufacturer';

  @override
  String get reviewExcellent => 'Excellent %';

  @override
  String get reviewAverage => 'Average %';

  @override
  String get reviewPoor => 'Poor %';

  @override
  String get validationUsernameRequired => 'Please enter your username.';

  @override
  String get validationPasswordRequired => 'Please enter your password.';

  @override
  String validationPasswordTooShort(int length) {
    return 'Password must be at least $length characters.';
  }

  @override
  String get validationEmailRequired => 'Please enter your email address.';

  @override
  String get validationEmailInvalid => 'That email address is not valid.';

  @override
  String get medicalDisclaimer => 'For reference only. This does not replace advice from a doctor or pharmacist.';

  @override
  String get chatTitle => 'Medicine Q&A';

  @override
  String get chatHint => 'Ask about any medicine…';

  @override
  String get chatSend => 'Send';

  @override
  String get chatStop => 'Stop';

  @override
  String get chatEmpty => 'Ask about any medicine — ingredients, uses, or side effects.';

  @override
  String get chatClear => 'Clear chat';

  @override
  String get chatClearTitle => 'Clear chat?';

  @override
  String get chatClearBody => 'All messages will be removed.';

  @override
  String chatToolRunning(String toolName) {
    return 'Looking up $toolName…';
  }

  @override
  String get healthProfileTitle => 'Health profile';

  @override
  String get healthConsentTitle => 'Consent to store profile';

  @override
  String get healthConsentBody => 'Your health profile (allergies, conditions, current medications) is stored on this device and is not uploaded; only the current-medications list is attached, temporarily, when you ask the assistant, to warn about interactions. You can delete it at any time.';

  @override
  String get healthConsentAccept => 'I agree';

  @override
  String get healthConsentGiven => 'Profile storage consent given';

  @override
  String get healthAllergies => 'Allergies';

  @override
  String get healthAllergiesHint => 'e.g. Penicillin, Aspirin';

  @override
  String get healthConditions => 'Conditions';

  @override
  String get healthConditionsHint => 'e.g. Diabetes, Hypertension';

  @override
  String get healthCurrentMeds => 'Current medications';

  @override
  String get healthCurrentMedsHint => 'e.g. Metformin, Amlodipine';

  @override
  String get healthSave => 'Save profile';

  @override
  String get healthSaved => 'Profile saved';

  @override
  String get healthDelete => 'Delete profile';

  @override
  String get healthDeleteTitle => 'Delete health profile?';

  @override
  String get healthDeleteBody => 'All allergies, conditions, medications and reminders will be removed.';

  @override
  String get reminderTitle => 'Medication reminders';

  @override
  String get reminderEmpty => 'No reminders yet';

  @override
  String get reminderAdd => 'Add reminder';

  @override
  String get reminderMedicineName => 'Medicine name';

  @override
  String get reminderDosage => 'Dosage';

  @override
  String get reminderDosageHint => 'e.g. 1 tablet, 5ml';

  @override
  String get consentTitle => 'Before you start';

  @override
  String get consentIntro => 'MedAssist helps you look up medicine information. Here is how your data is handled:';

  @override
  String get consentPointLocal => 'Your health profile, reminders and intake log are stored only on this device.';

  @override
  String get consentPointServer => 'Search keywords, questions to the assistant and photos you choose to identify (in server mode) are sent to the MedAssist server. Nothing is shared with third parties.';

  @override
  String get consentPointMedical => 'Information is for reference only and does not replace a doctor or pharmacist.';

  @override
  String get consentPointDelete => 'You can delete your data and account at any time under Privacy.';

  @override
  String get consentAccept => 'Agree and continue';

  @override
  String get consentDecline => 'Decline';

  @override
  String get privacyTitle => 'Privacy';

  @override
  String get privacySummary => 'Your health profile and reminders stay on this device. Questions and search keywords go to the MedAssist server and are not shared with third parties. Signing out clears saved offline data. Medicine and condition names used for \"Tips\" are sent to the MedAssist server for that request only (not stored, not logged), separately from cloud AI.';

  @override
  String get privacyRevokeConsent => 'Withdraw consent';

  @override
  String get privacyRevokeConsentBody => 'You will be asked again the next time you open the app.';

  @override
  String get privacyDeleteAccount => 'Delete account and data';

  @override
  String get deleteAccountTitle => 'Delete account?';

  @override
  String get deleteAccountBody => 'Permanently deletes your account, history and conversations on the server, plus the health profile and reminders on this device. This cannot be undone.';

  @override
  String get deleteAccountPasswordLabel => 'Enter your password to confirm';

  @override
  String get deleteAccountConfirm => 'Delete permanently';

  @override
  String get deleteAccountDone => 'Account and data deleted.';

  @override
  String get chatAiLabel => 'AI assistant · answers only from loaded sources';

  @override
  String get navHome => 'Home';

  @override
  String get navReminders => 'Reminders';

  @override
  String get navAssistant => 'Assistant';

  @override
  String get navProfile => 'Profile';

  @override
  String get scanMedicine => 'Scan medicine';

  @override
  String get homeGreeting => 'Hello!';

  @override
  String get homeSubtitle => 'Here is your medicine for today';

  @override
  String get homeTodayTitle => 'Today\'s medicines';

  @override
  String get homeNoReminders => 'No reminders yet. Add one under Reminders.';

  @override
  String get homeSeeAll => 'See all';

  @override
  String get homeSearchHint => 'Search medicine by name…';

  @override
  String get homeShortcuts => 'Shortcuts';

  @override
  String get reminderSetAlarm => 'Set your alarm';

  @override
  String get reminderSave => 'Save';

  @override
  String get reminderEveryDay => 'Every day';

  @override
  String get reminderTimeLabel => 'Time';

  @override
  String get voiceMicTooltip => 'Speak to the assistant';

  @override
  String get voiceListening => 'Listening…';

  @override
  String get voiceUnavailable => 'Voice is not enabled';

  @override
  String get profileTitle => 'My profile';

  @override
  String get profileAccountSection => 'Account';

  @override
  String get profileDataSection => 'Data & privacy';

  @override
  String homeGreetingNamed(String name) {
    return 'Hello, $name!';
  }

  @override
  String get homeMarkTaken => 'Mark as taken';

  @override
  String get homeMarkedTaken => 'Taken today';

  @override
  String get doseASkip => 'Skip';

  @override
  String get doseASnooze => 'Snooze 15 min';

  @override
  String get doseASkippedToday => 'Skipped today';

  @override
  String get doseANextDose => 'Next dose';

  @override
  String get doseAMorning => 'Morning';

  @override
  String get doseANoon => 'Noon';

  @override
  String get doseAAfternoon => 'Afternoon';

  @override
  String get doseAEvening => 'Evening';

  @override
  String get doseASnoozedMsg => 'Will remind you again in 15 minutes';

  @override
  String get progressShortcut => 'Progress';

  @override
  String get progressTitle => 'Adherence progress';

  @override
  String progressStreakDays(int count) {
    return '$count-day streak!';
  }

  @override
  String get progressStreakZero => 'Take your medicine today to start a streak';

  @override
  String get progressWeekTitle => 'Last 7 days';

  @override
  String get progressEmpty => 'No doses logged yet.';

  @override
  String get reminderDuplicateWarning => 'This looks like a medicine you already have in reminders or your health profile — double-check before saving.';

  @override
  String get lockSectionTitle => 'Lock app with a PIN';

  @override
  String get lockSectionSubtitle => 'Require a PIN when reopening the app after being away for a while.';

  @override
  String get lockEnterPinTitle => 'Enter your PIN to unlock';

  @override
  String get lockUnlock => 'Unlock';

  @override
  String get lockPinWrong => 'Wrong PIN, try again.';

  @override
  String get lockSetNewPinTitle => 'Set a new PIN';

  @override
  String get lockPinLabel => 'PIN (4-6 digits)';

  @override
  String get lockPinConfirmLabel => 'Confirm PIN';

  @override
  String get lockPinTooShort => 'PIN must be at least 4 digits.';

  @override
  String get lockPinMismatch => 'The two PINs don\'t match.';

  @override
  String get lockDisableTitle => 'Enter your current PIN to turn off the lock';

  @override
  String get lockSave => 'Save';

  @override
  String get lockEnabled => 'PIN lock enabled';

  @override
  String get lockDisabled => 'PIN lock disabled';

  @override
  String get newsLeaveTitle => 'Leave the app to read this?';

  @override
  String newsLeaveBody(String host) {
    return 'The article opens in your browser at $host. That site is not run by MedAssist and can see your IP address. No health data is sent.';
  }

  @override
  String newsVerifiedSources(String domains) {
    return 'Links only from verified sources: $domains';
  }

  @override
  String get newsLeaveOpen => 'Open browser';

  @override
  String get newsLeaveCopy => 'Copy link';

  @override
  String get rxCreateFromScan => 'Create reminders from prescription';

  @override
  String get rxNothingFound => 'No medicines found in this prescription. Please add reminders manually.';

  @override
  String get rxReviewTitle => 'Confirm prescription';

  @override
  String get rxAiBanner => 'AI read this prescription — check everything before saving; handwritten prescriptions may be misread.';

  @override
  String get rxInclude => 'Add to reminders';

  @override
  String get rxMedicineName => 'Medicine name';

  @override
  String get rxDose => 'Dose per time';

  @override
  String get rxTimes => 'Times';

  @override
  String get rxAddTime => 'Add time';

  @override
  String get rxMealRelation => 'With meals';

  @override
  String get rxMealNone => 'Not specified';

  @override
  String get rxMealBefore => 'Before meals';

  @override
  String get rxMealAfter => 'After meals';

  @override
  String get rxMealDuring => 'During meals';

  @override
  String get rxNeedsReview => 'Check this: frequency could not be read clearly';

  @override
  String get rxDuplicate => 'This medicine is already in your reminders';

  @override
  String get rxDiagnosisTitle => 'Diagnosis (read from prescription)';

  @override
  String get rxAddDiagnosisToProfile => 'Add to health profile';

  @override
  String get rxProfileConsentRequired => 'Health profile consent is required before adding the diagnosis or medicines to your profile.';

  @override
  String get rxSave => 'Save and create reminders';

  @override
  String rxSaved(int count) {
    return 'Created $count reminders';
  }

  @override
  String get rxNoItemSelected => 'Select at least one medicine with a name and a time.';

  @override
  String get rxSaveFailed => 'Could not save the prescription. Please try again.';

  @override
  String get facilityTitle => 'Nearest medical facility';

  @override
  String get facilityCall115 => 'Call emergency 115';

  @override
  String get facilitySampleBanner => 'Sample data, needs verification — confirm the address before going.';

  @override
  String get facilityLocationRationale => 'Your location is used only to calculate distances on this phone. It is never sent anywhere.';

  @override
  String get facilityUseLocation => 'Use my location';

  @override
  String get facilityLocating => 'Getting your location…';

  @override
  String get facilityPermissionDenied => 'Location permission was denied. The list is not sorted by distance; you can grant permission in Settings.';

  @override
  String get facilityLocationOff => 'Location is turned off. Turn on Location and try again.';

  @override
  String get facilityLocationUnavailable => 'Could not get your location. Try again or search on the map.';

  @override
  String get facilityUnsupported => 'This device does not support location features.';

  @override
  String get facilityDirections => 'Directions';

  @override
  String get facilitySearchMore => 'Search more on the map';

  @override
  String get facilitySearchQuery => 'hospitals near me';

  @override
  String facilityDistanceKm(String km) {
    return '$km km away';
  }

  @override
  String get facilityOpenFailed => 'No app on this phone can handle this action.';

  @override
  String get facilityLoadFailed => 'Could not load the medical facility list.';

  @override
  String get cloudAiTitle => 'Cloud AI (Claude)';

  @override
  String get cloudAiSubtitle => 'Assistant answers more naturally using Anthropic\'s AI. Off by default.';

  @override
  String get cloudAiDialogTitle => 'Use cloud AI?';

  @override
  String get cloudAiDialogBody => 'Recipient: Anthropic (Claude, cloud AI).\n\nSent: your question (with identifying details such as name, phone, email, dates and document numbers removed), the medicines you currently take, and the conditions/allergies you entered in your profile.\n\nNote: for \"Tips\", medicine and condition names go to the MedAssist server for that request only (not stored, not logged) - separate from cloud AI.\n\nNot sent: prescription photos, location, your full profile.\n\nYou can turn this off at any time. Information is for reference only and is not a diagnosis.';

  @override
  String get cloudAiAgree => 'Agree';

  @override
  String get cloudAiDecline => 'No';

  @override
  String cloudAiGrantedAt(String when) {
    return 'Agreed on $when';
  }

  @override
  String get cloudAiChatHint => 'Turn on \'Cloud AI\' in Profile > Privacy for more natural answers';

  @override
  String get cloudAiHintDismiss => 'Dismiss hint';

  @override
  String get emergencyCall => 'Call 115';

  @override
  String get emergencyNearest => 'Nearest medical facility';

  @override
  String get emergencyDialFailed => 'Could not open the dialer. Please call 115 manually.';

  @override
  String get lifestyleTitle => 'Tips for you';

  @override
  String get lifestyleEmpty => 'No tips right now.';

  @override
  String get lifestyleError => 'Could not load tips.';

  @override
  String get lifestyleRefresh => 'Refresh tips';

  @override
  String lifestyleSource(String name) {
    return 'Source: $name';
  }

  @override
  String get lifestyleAiBadge => 'AI';

  @override
  String get polishDViewMedicine => 'View uses & side effects';

  @override
  String polishDQuoteLabel(String id) {
    return 'Excerpt ($id)';
  }

  @override
  String get stockBPillsTotal => 'Pills dispensed (optional)';

  @override
  String get stockBPillsLeft => 'Pills left (optional)';

  @override
  String get stockBPillsPerDose => 'Pills per dose';

  @override
  String get stockBConfirmed => 'I have checked this';

  @override
  String get stockBNeedsCheck => 'Please check this item against the prescription and tick the box.';

  @override
  String stockBLowStock(String name, int count) {
    return 'Running low: $name has $count pills left. Buy more or ask a pharmacist.';
  }

  @override
  String get stockBEmptyReminderTitle => 'No reminders yet';

  @override
  String get stockBScanPrescription => 'Scan prescription';

  @override
  String get reportCTitle => 'Adherence report';

  @override
  String get reportCRange7 => '7 days';

  @override
  String get reportCRange30 => '30 days';

  @override
  String get reportCRate => 'On-time rate';

  @override
  String get reportCTaken => 'Taken';

  @override
  String get reportCSkipped => 'Skipped';

  @override
  String get reportCLate => 'Late';

  @override
  String get reportCSnoozed => 'Snoozed';

  @override
  String get reportCStreakNow => 'Current streak';

  @override
  String get reportCStreakBest => 'Best streak';

  @override
  String get reportCDays => 'days';

  @override
  String get reportCChartTitle => 'Doses taken per day';

  @override
  String get reportCNoRate => 'No data';

  @override
  String get reportCNote => 'For reference only; talk to your doctor or pharmacist. Do not change or stop medicine on your own.';

  @override
  String get reportCDoctorMode => 'Show to doctor';

  @override
  String get reportCDoctorTitle => 'Summary for doctor';

  @override
  String get reportCBack => 'Back';

  @override
  String get reportCEmpty => 'No doses logged yet. Mark doses as taken to see your report.';

  @override
  String get reportCDoctorHint => 'Read-only summary stored only on this phone.';

  @override
  String get reportCPeriod => 'Period';

  @override
  String get fixELifestyleNote => 'For \"Tips\", medicine and condition names are sent to the MedAssist server for that request only (not stored, not logged). This is separate from sending data to cloud AI.';

  @override
  String fixELockTooMany(String time) {
    return 'Too many wrong attempts. Try again in $time.';
  }

  @override
  String get fixELockForgot => 'Forgot PIN? Sign out';

  @override
  String get fixELockUnreadable => 'Secure storage cannot be read, so the app stays locked. Sign out to reset the PIN.';

  @override
  String get modeGUnverified => 'Sample data - not verified';

  @override
  String modeGUnmatched(String conditions) {
    return 'No verified nutrition document for: $conditions. Please ask a doctor or a dietitian.';
  }

  @override
  String get modeGChipExtractive => 'Quoted from loaded sources';

  @override
  String get modeGChipCloud => 'Reworded by Claude - still source-based';

  @override
  String get modeGSheetTitle => 'Two answer modes';

  @override
  String get modeGOff => 'Cloud AI off';

  @override
  String get modeGOn => 'Cloud AI on';

  @override
  String get modeGRowData => 'Does data leave the MedAssist server?';

  @override
  String get modeGOffData => 'No. Your question is handled on the MedAssist server only, nothing goes to a third party.';

  @override
  String get modeGOnData => 'Yes. Your question (identifying details removed) plus the medicines/conditions you agreed to share go to Anthropic (Claude).';

  @override
  String get modeGRowHow => 'How it answers';

  @override
  String get modeGOffHow => 'Quotes the loaded documents verbatim.';

  @override
  String get modeGOnHow => 'Claude rewords it more naturally using only the loaded documents; if it strays from the sources, the verbatim quote is used instead.';

  @override
  String get modeGRowSame => 'What does NOT change';

  @override
  String get modeGSameBody => 'Sources are always cited; no diagnosis or dosing; emergency warnings still appear.';

  @override
  String get modeGWhere => 'Want to switch? Go to Profile > Privacy > Cloud AI.';

  @override
  String get modeGClose => 'Got it';

  @override
  String get modeGConsentModes => 'Off: answers quote the loaded documents verbatim and nothing is sent to a third party. On: Claude (Anthropic) rewords answers more naturally but must still follow the sources.';

  @override
  String get memHTitle => 'What the assistant remembers about me';

  @override
  String get memHSubtitle => 'See, turn off or delete what the assistant remembers - stored on this device only.';

  @override
  String get memHToggle => 'Let the assistant remember my symptoms';

  @override
  String get memHToggleBody => 'Stored ONLY on this device, encrypted, and you can delete it any time. Each question sends only the minimum context to the MedAssist server, which does not store it.';

  @override
  String get memHNeedProfile => 'Turn on the Health profile (consent to store) first.';

  @override
  String get memHUnreadable => 'Secure data cannot be read';

  @override
  String get memHUnreadableBody => 'The encrypted data on this device cannot be read. The app will not overwrite it. You can delete all assistant data to start over.';

  @override
  String get memHStoredTitle => 'Stored on this device';

  @override
  String memHConditions(int count) {
    return 'Conditions: $count';
  }

  @override
  String memHAllergies(int count) {
    return 'Allergies: $count';
  }

  @override
  String memHMeds(int count) {
    return 'Current medications: $count';
  }

  @override
  String memHEpisodes(int count) {
    return 'Symptom log: $count entries';
  }

  @override
  String get memHNone => 'None';

  @override
  String get memHEpisodesTitle => 'Last 20 entries';

  @override
  String get memHRecurringTag => 'recurring';

  @override
  String get memHPreviewTitle => 'What the assistant will use for your next question';

  @override
  String get memHPreviewConditions => 'Conditions';

  @override
  String get memHPreviewAllergies => 'Allergies';

  @override
  String get memHPreviewSymptoms => 'Recent symptoms';

  @override
  String get memHPreviewNothing => 'Nothing is sent besides your question.';

  @override
  String get memHCloudYes => 'Cloud AI: receives only the de-identified question (you consented separately).';

  @override
  String get memHCloudNo => 'Cloud AI: receives nothing (no consent given).';

  @override
  String get memHClearLog => 'Delete symptom log';

  @override
  String get memHClearAll => 'Delete all assistant data';

  @override
  String get memHClearLogConfirm => 'Delete the whole symptom log on this device?';

  @override
  String get memHClearAllConfirm => 'Delete the health profile, symptom log and turn off assistant memory on this device? This cannot be undone.';

  @override
  String get memHCleared => 'Deleted';

  @override
  String memHChatSaved(String terms) {
    return 'Remembered on this device: $terms';
  }

  @override
  String get memHChatForget => 'Delete';

  @override
  String get memHRecurring => 'Recurring symptom';

  @override
  String get srcJTitle => 'Source';

  @override
  String srcJRetrieved(String date) {
    return 'Retrieved: $date';
  }

  @override
  String srcJLicense(String text) {
    return 'License: $text';
  }

  @override
  String srcJReviewed(String name) {
    return 'Reviewed by: $name';
  }

  @override
  String get srcJNotReviewed => 'No reviewer yet';

  @override
  String get srcJOpen => 'Open original source';

  @override
  String srcJOpenHost(String host) {
    return 'Open original source ($host)';
  }

  @override
  String get srcJNoLink => 'No original link';

  @override
  String get srcJVerified => 'Verified source';

  @override
  String get rxCheckCatalog => 'Check against medicine list';

  @override
  String get rxCheckCatalogNote => 'Sends the medicine name to the MedAssist server to look it up in the list (not stored, not logged).';

  @override
  String get rxCatalogExact => 'Found in the medicine list';

  @override
  String get rxCatalogSimilar => 'No exact match — see suggestions below and check the spelling';

  @override
  String get rxCatalogNone => 'Not found in the medicine list — OCR may have misread it; compare with the prescription';

  @override
  String get rxCatalogError => 'Could not check (offline?). Please verify manually.';

  @override
  String get rxUseThisName => 'Use this name';

  @override
  String rxCatalogStrength(String value) {
    return 'Ingredient · strength: $value';
  }

  @override
  String rxLowConfidence(String reasons) {
    return 'Low confidence: $reasons';
  }

  @override
  String get rxIssueFrequency => 'frequency unclear';

  @override
  String get rxIssueDose => 'dose missing';

  @override
  String get rxIssueShortName => 'name too short';

  @override
  String get rxIssueOddChars => 'name has unusual characters';

  @override
  String get rxIssueNotInCatalog => 'not in the medicine list';

  @override
  String get displaySection => 'Display';

  @override
  String get displayTheme => 'Theme';

  @override
  String get displayThemeSystem => 'Device';

  @override
  String get displayThemeLight => 'Light';

  @override
  String get displayThemeDark => 'Dark';

  @override
  String get displayTextSize => 'Text size';

  @override
  String get displayTextNormal => 'Normal';

  @override
  String get displayTextLarge => 'Large';

  @override
  String get displayTextExtraLarge => 'Extra large';

  @override
  String get doseASnoozeMenu => 'Snooze…';

  @override
  String doseASnoozeFor(int minutes) {
    return 'Snooze $minutes min';
  }

  @override
  String doseASnoozedFor(int minutes) {
    return 'Will remind you again in $minutes minutes';
  }
}
