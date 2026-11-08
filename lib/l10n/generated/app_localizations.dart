import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:intl/intl.dart' as intl;

import 'app_localizations_en.dart';
import 'app_localizations_vi.dart';

/// Callers can lookup localized strings with an instance of AppLocalizations
/// returned by `AppLocalizations.of(context)`.
///
/// Applications need to include `AppLocalizations.delegate()` in their app's
/// `localizationDelegates` list, and the locales they support in the app's
/// `supportedLocales` list. For example:
///
/// ```dart
/// import 'generated/app_localizations.dart';
///
/// return MaterialApp(
///   localizationsDelegates: AppLocalizations.localizationsDelegates,
///   supportedLocales: AppLocalizations.supportedLocales,
///   home: MyApplicationHome(),
/// );
/// ```
///
/// ## Update pubspec.yaml
///
/// Please make sure to update your pubspec.yaml to include the following
/// packages:
///
/// ```yaml
/// dependencies:
///   # Internationalization support.
///   flutter_localizations:
///     sdk: flutter
///   intl: any # Use the pinned version from flutter_localizations
///
///   # Rest of dependencies
/// ```
///
/// ## iOS Applications
///
/// iOS applications define key application metadata, including supported
/// locales, in an Info.plist file that is built into the application bundle.
/// To configure the locales supported by your app, you’ll need to edit this
/// file.
///
/// First, open your project’s ios/Runner.xcworkspace Xcode workspace file.
/// Then, in the Project Navigator, open the Info.plist file under the Runner
/// project’s Runner folder.
///
/// Next, select the Information Property List item, select Add Item from the
/// Editor menu, then select Localizations from the pop-up menu.
///
/// Select and expand the newly-created Localizations item then, for each
/// locale your application supports, add a new item and select the locale
/// you wish to add from the pop-up menu in the Value field. This list should
/// be consistent with the languages listed in the AppLocalizations.supportedLocales
/// property.
abstract class AppLocalizations {
  AppLocalizations(String locale) : localeName = intl.Intl.canonicalizedLocale(locale.toString());

  final String localeName;

  static AppLocalizations of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations)!;
  }

  static const LocalizationsDelegate<AppLocalizations> delegate = _AppLocalizationsDelegate();

  /// A list of this localizations delegate along with the default localizations
  /// delegates.
  ///
  /// Returns a list of localizations delegates containing this delegate along with
  /// GlobalMaterialLocalizations.delegate, GlobalCupertinoLocalizations.delegate,
  /// and GlobalWidgetsLocalizations.delegate.
  ///
  /// Additional delegates can be added by appending to this list in
  /// MaterialApp. This list does not have to be used at all if a custom list
  /// of delegates is preferred or required.
  static const List<LocalizationsDelegate<dynamic>> localizationsDelegates = <LocalizationsDelegate<dynamic>>[
    delegate,
    GlobalMaterialLocalizations.delegate,
    GlobalCupertinoLocalizations.delegate,
    GlobalWidgetsLocalizations.delegate,
  ];

  /// A list of this localizations delegate's supported locales.
  static const List<Locale> supportedLocales = <Locale>[
    Locale('vi'),
    Locale('en')
  ];

  /// Application name shown in the task switcher and app bar.
  ///
  /// In en, this message translates to:
  /// **'MedAssist'**
  String get appTitle;

  /// No description provided for @loginTitle.
  ///
  /// In en, this message translates to:
  /// **'Sign in'**
  String get loginTitle;

  /// No description provided for @loginSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Medicine information you can rely on'**
  String get loginSubtitle;

  /// No description provided for @usernameLabel.
  ///
  /// In en, this message translates to:
  /// **'Username'**
  String get usernameLabel;

  /// No description provided for @passwordLabel.
  ///
  /// In en, this message translates to:
  /// **'Password'**
  String get passwordLabel;

  /// No description provided for @emailLabel.
  ///
  /// In en, this message translates to:
  /// **'Email'**
  String get emailLabel;

  /// No description provided for @signIn.
  ///
  /// In en, this message translates to:
  /// **'Sign in'**
  String get signIn;

  /// No description provided for @signUp.
  ///
  /// In en, this message translates to:
  /// **'Sign up'**
  String get signUp;

  /// No description provided for @signUpTitle.
  ///
  /// In en, this message translates to:
  /// **'Create an account'**
  String get signUpTitle;

  /// No description provided for @signUpLink.
  ///
  /// In en, this message translates to:
  /// **'Create an account'**
  String get signUpLink;

  /// No description provided for @showPassword.
  ///
  /// In en, this message translates to:
  /// **'Show password'**
  String get showPassword;

  /// No description provided for @hidePassword.
  ///
  /// In en, this message translates to:
  /// **'Hide password'**
  String get hidePassword;

  /// No description provided for @signInFailed.
  ///
  /// In en, this message translates to:
  /// **'Sign-in failed.'**
  String get signInFailed;

  /// No description provided for @signUpFailed.
  ///
  /// In en, this message translates to:
  /// **'Sign-up failed.'**
  String get signUpFailed;

  /// No description provided for @processFailed.
  ///
  /// In en, this message translates to:
  /// **'Could not analyse the image.'**
  String get processFailed;

  /// No description provided for @createAccount.
  ///
  /// In en, this message translates to:
  /// **'Create an account'**
  String get createAccount;

  /// No description provided for @signOut.
  ///
  /// In en, this message translates to:
  /// **'Sign out'**
  String get signOut;

  /// No description provided for @signOutConfirmTitle.
  ///
  /// In en, this message translates to:
  /// **'Sign out?'**
  String get signOutConfirmTitle;

  /// No description provided for @signOutConfirmBody.
  ///
  /// In en, this message translates to:
  /// **'You will need to sign in again to continue.'**
  String get signOutConfirmBody;

  /// No description provided for @cancel.
  ///
  /// In en, this message translates to:
  /// **'Cancel'**
  String get cancel;

  /// No description provided for @confirm.
  ///
  /// In en, this message translates to:
  /// **'Confirm'**
  String get confirm;

  /// No description provided for @delete.
  ///
  /// In en, this message translates to:
  /// **'Delete'**
  String get delete;

  /// No description provided for @processImageTitle.
  ///
  /// In en, this message translates to:
  /// **'Scan medicine'**
  String get processImageTitle;

  /// No description provided for @pickImage.
  ///
  /// In en, this message translates to:
  /// **'Pick image'**
  String get pickImage;

  /// No description provided for @processImage.
  ///
  /// In en, this message translates to:
  /// **'Analyse'**
  String get processImage;

  /// No description provided for @noImagePicked.
  ///
  /// In en, this message translates to:
  /// **'No image selected yet'**
  String get noImagePicked;

  /// No description provided for @history.
  ///
  /// In en, this message translates to:
  /// **'History'**
  String get history;

  /// No description provided for @search.
  ///
  /// In en, this message translates to:
  /// **'Search'**
  String get search;

  /// No description provided for @resultsTitle.
  ///
  /// In en, this message translates to:
  /// **'Results:'**
  String get resultsTitle;

  /// No description provided for @noResults.
  ///
  /// In en, this message translates to:
  /// **'No results.'**
  String get noResults;

  /// No description provided for @processSuccess.
  ///
  /// In en, this message translates to:
  /// **'Done!'**
  String get processSuccess;

  /// No description provided for @searchTitle.
  ///
  /// In en, this message translates to:
  /// **'Search medicine'**
  String get searchTitle;

  /// No description provided for @searchHint.
  ///
  /// In en, this message translates to:
  /// **'Medicine name'**
  String get searchHint;

  /// No description provided for @searchEmptyQuery.
  ///
  /// In en, this message translates to:
  /// **'Please enter a medicine name.'**
  String get searchEmptyQuery;

  /// No description provided for @historyTitle.
  ///
  /// In en, this message translates to:
  /// **'Processing history'**
  String get historyTitle;

  /// No description provided for @historyEmpty.
  ///
  /// In en, this message translates to:
  /// **'No processing history'**
  String get historyEmpty;

  /// No description provided for @historyLoadFailed.
  ///
  /// In en, this message translates to:
  /// **'Could not load history.'**
  String get historyLoadFailed;

  /// No description provided for @searchResultsLabel.
  ///
  /// In en, this message translates to:
  /// **'Search results:'**
  String get searchResultsLabel;

  /// No description provided for @historyFile.
  ///
  /// In en, this message translates to:
  /// **'File:'**
  String get historyFile;

  /// No description provided for @deleteRecordTitle.
  ///
  /// In en, this message translates to:
  /// **'Delete this record?'**
  String get deleteRecordTitle;

  /// No description provided for @deleteRecordBody.
  ///
  /// In en, this message translates to:
  /// **'This record will be removed from your history.'**
  String get deleteRecordBody;

  /// No description provided for @deleted.
  ///
  /// In en, this message translates to:
  /// **'Deleted'**
  String get deleted;

  /// No description provided for @deleteFailed.
  ///
  /// In en, this message translates to:
  /// **'Could not delete the record'**
  String get deleteFailed;

  /// No description provided for @retry.
  ///
  /// In en, this message translates to:
  /// **'Retry'**
  String get retry;

  /// No description provided for @newsTitle.
  ///
  /// In en, this message translates to:
  /// **'Medicine & health news'**
  String get newsTitle;

  /// No description provided for @newsEmpty.
  ///
  /// In en, this message translates to:
  /// **'No news yet.'**
  String get newsEmpty;

  /// No description provided for @newsLoadFailed.
  ///
  /// In en, this message translates to:
  /// **'Could not load news.'**
  String get newsLoadFailed;

  /// No description provided for @newsSampleBanner.
  ///
  /// In en, this message translates to:
  /// **'Showing sample content — a real news source isn\'t enabled yet.'**
  String get newsSampleBanner;

  /// No description provided for @newsOpenSource.
  ///
  /// In en, this message translates to:
  /// **'View source'**
  String get newsOpenSource;

  /// No description provided for @newsNoSourcesConfigured.
  ///
  /// In en, this message translates to:
  /// **'The project owner hasn\'t enabled a news source yet. See docs/NEWS.md.'**
  String get newsNoSourcesConfigured;

  /// No description provided for @offline.
  ///
  /// In en, this message translates to:
  /// **'Offline'**
  String get offline;

  /// Banner shown when showing cached data because the network failed.
  ///
  /// In en, this message translates to:
  /// **'Offline · updated {time}'**
  String offlineUpdated(String time);

  /// No description provided for @noInformation.
  ///
  /// In en, this message translates to:
  /// **'No information'**
  String get noInformation;

  /// No description provided for @medicineName.
  ///
  /// In en, this message translates to:
  /// **'Medicine name'**
  String get medicineName;

  /// No description provided for @composition.
  ///
  /// In en, this message translates to:
  /// **'Ingredients'**
  String get composition;

  /// No description provided for @uses.
  ///
  /// In en, this message translates to:
  /// **'Uses'**
  String get uses;

  /// No description provided for @sideEffects.
  ///
  /// In en, this message translates to:
  /// **'Side effects'**
  String get sideEffects;

  /// No description provided for @manufacturer.
  ///
  /// In en, this message translates to:
  /// **'Manufacturer'**
  String get manufacturer;

  /// No description provided for @reviewExcellent.
  ///
  /// In en, this message translates to:
  /// **'Excellent %'**
  String get reviewExcellent;

  /// No description provided for @reviewAverage.
  ///
  /// In en, this message translates to:
  /// **'Average %'**
  String get reviewAverage;

  /// No description provided for @reviewPoor.
  ///
  /// In en, this message translates to:
  /// **'Poor %'**
  String get reviewPoor;

  /// No description provided for @validationUsernameRequired.
  ///
  /// In en, this message translates to:
  /// **'Please enter your username.'**
  String get validationUsernameRequired;

  /// No description provided for @validationPasswordRequired.
  ///
  /// In en, this message translates to:
  /// **'Please enter your password.'**
  String get validationPasswordRequired;

  /// No description provided for @validationPasswordTooShort.
  ///
  /// In en, this message translates to:
  /// **'Password must be at least {length} characters.'**
  String validationPasswordTooShort(int length);

  /// No description provided for @validationEmailRequired.
  ///
  /// In en, this message translates to:
  /// **'Please enter your email address.'**
  String get validationEmailRequired;

  /// No description provided for @validationEmailInvalid.
  ///
  /// In en, this message translates to:
  /// **'That email address is not valid.'**
  String get validationEmailInvalid;

  /// Mandatory medical disclaimer shown on result and chat screens. Never fabricate sources or medical claims.
  ///
  /// In en, this message translates to:
  /// **'For reference only. This does not replace advice from a doctor or pharmacist.'**
  String get medicalDisclaimer;

  /// No description provided for @chatTitle.
  ///
  /// In en, this message translates to:
  /// **'Medicine Q&A'**
  String get chatTitle;

  /// No description provided for @chatHint.
  ///
  /// In en, this message translates to:
  /// **'Ask about any medicine…'**
  String get chatHint;

  /// No description provided for @chatSend.
  ///
  /// In en, this message translates to:
  /// **'Send'**
  String get chatSend;

  /// No description provided for @chatStop.
  ///
  /// In en, this message translates to:
  /// **'Stop'**
  String get chatStop;

  /// No description provided for @chatEmpty.
  ///
  /// In en, this message translates to:
  /// **'Ask about any medicine — ingredients, uses, or side effects.'**
  String get chatEmpty;

  /// No description provided for @chatClear.
  ///
  /// In en, this message translates to:
  /// **'Clear chat'**
  String get chatClear;

  /// No description provided for @chatClearTitle.
  ///
  /// In en, this message translates to:
  /// **'Clear chat?'**
  String get chatClearTitle;

  /// No description provided for @chatClearBody.
  ///
  /// In en, this message translates to:
  /// **'All messages will be removed.'**
  String get chatClearBody;

  /// No description provided for @chatToolRunning.
  ///
  /// In en, this message translates to:
  /// **'Looking up {toolName}…'**
  String chatToolRunning(String toolName);

  /// No description provided for @healthProfileTitle.
  ///
  /// In en, this message translates to:
  /// **'Health profile'**
  String get healthProfileTitle;

  /// No description provided for @healthConsentTitle.
  ///
  /// In en, this message translates to:
  /// **'Consent to store profile'**
  String get healthConsentTitle;

  /// No description provided for @healthConsentBody.
  ///
  /// In en, this message translates to:
  /// **'Your health profile (allergies, conditions, current medications) is stored on this device and is not uploaded; only the current-medications list is attached, temporarily, when you ask the assistant, to warn about interactions. You can delete it at any time.'**
  String get healthConsentBody;

  /// No description provided for @healthConsentAccept.
  ///
  /// In en, this message translates to:
  /// **'I agree'**
  String get healthConsentAccept;

  /// No description provided for @healthConsentGiven.
  ///
  /// In en, this message translates to:
  /// **'Profile storage consent given'**
  String get healthConsentGiven;

  /// No description provided for @healthAllergies.
  ///
  /// In en, this message translates to:
  /// **'Allergies'**
  String get healthAllergies;

  /// No description provided for @healthAllergiesHint.
  ///
  /// In en, this message translates to:
  /// **'e.g. Penicillin, Aspirin'**
  String get healthAllergiesHint;

  /// No description provided for @healthConditions.
  ///
  /// In en, this message translates to:
  /// **'Conditions'**
  String get healthConditions;

  /// No description provided for @healthConditionsHint.
  ///
  /// In en, this message translates to:
  /// **'e.g. Diabetes, Hypertension'**
  String get healthConditionsHint;

  /// No description provided for @healthCurrentMeds.
  ///
  /// In en, this message translates to:
  /// **'Current medications'**
  String get healthCurrentMeds;

  /// No description provided for @healthCurrentMedsHint.
  ///
  /// In en, this message translates to:
  /// **'e.g. Metformin, Amlodipine'**
  String get healthCurrentMedsHint;

  /// No description provided for @healthSave.
  ///
  /// In en, this message translates to:
  /// **'Save profile'**
  String get healthSave;

  /// No description provided for @healthSaved.
  ///
  /// In en, this message translates to:
  /// **'Profile saved'**
  String get healthSaved;

  /// No description provided for @healthDelete.
  ///
  /// In en, this message translates to:
  /// **'Delete profile'**
  String get healthDelete;

  /// No description provided for @healthDeleteTitle.
  ///
  /// In en, this message translates to:
  /// **'Delete health profile?'**
  String get healthDeleteTitle;

  /// No description provided for @healthDeleteBody.
  ///
  /// In en, this message translates to:
  /// **'All allergies, conditions, medications and reminders will be removed.'**
  String get healthDeleteBody;

  /// No description provided for @reminderTitle.
  ///
  /// In en, this message translates to:
  /// **'Medication reminders'**
  String get reminderTitle;

  /// No description provided for @reminderEmpty.
  ///
  /// In en, this message translates to:
  /// **'No reminders yet'**
  String get reminderEmpty;

  /// No description provided for @reminderAdd.
  ///
  /// In en, this message translates to:
  /// **'Add reminder'**
  String get reminderAdd;

  /// No description provided for @reminderMedicineName.
  ///
  /// In en, this message translates to:
  /// **'Medicine name'**
  String get reminderMedicineName;

  /// No description provided for @reminderDosage.
  ///
  /// In en, this message translates to:
  /// **'Dosage'**
  String get reminderDosage;

  /// No description provided for @reminderDosageHint.
  ///
  /// In en, this message translates to:
  /// **'e.g. 1 tablet, 5ml'**
  String get reminderDosageHint;

  /// No description provided for @consentTitle.
  ///
  /// In en, this message translates to:
  /// **'Before you start'**
  String get consentTitle;

  /// No description provided for @consentIntro.
  ///
  /// In en, this message translates to:
  /// **'MedAssist helps you look up medicine information. Here is how your data is handled:'**
  String get consentIntro;

  /// No description provided for @consentPointLocal.
  ///
  /// In en, this message translates to:
  /// **'Your health profile, reminders and intake log are stored only on this device.'**
  String get consentPointLocal;

  /// No description provided for @consentPointServer.
  ///
  /// In en, this message translates to:
  /// **'Search keywords, questions to the assistant and photos you choose to identify (in server mode) are sent to the MedAssist server. Nothing is shared with third parties.'**
  String get consentPointServer;

  /// No description provided for @consentPointMedical.
  ///
  /// In en, this message translates to:
  /// **'Information is for reference only and does not replace a doctor or pharmacist.'**
  String get consentPointMedical;

  /// No description provided for @consentPointDelete.
  ///
  /// In en, this message translates to:
  /// **'You can delete your data and account at any time under Privacy.'**
  String get consentPointDelete;

  /// No description provided for @consentAccept.
  ///
  /// In en, this message translates to:
  /// **'Agree and continue'**
  String get consentAccept;

  /// No description provided for @consentDecline.
  ///
  /// In en, this message translates to:
  /// **'Decline'**
  String get consentDecline;

  /// No description provided for @privacyTitle.
  ///
  /// In en, this message translates to:
  /// **'Privacy'**
  String get privacyTitle;

  /// No description provided for @privacySummary.
  ///
  /// In en, this message translates to:
  /// **'Your health profile and reminders stay on this device. Questions and search keywords go to the MedAssist server and are not shared with third parties. Signing out clears saved offline data. Medicine and condition names used for \"Tips\" are sent to the MedAssist server for that request only (not stored, not logged), separately from cloud AI.'**
  String get privacySummary;

  /// No description provided for @privacyRevokeConsent.
  ///
  /// In en, this message translates to:
  /// **'Withdraw consent'**
  String get privacyRevokeConsent;

  /// No description provided for @privacyRevokeConsentBody.
  ///
  /// In en, this message translates to:
  /// **'You will be asked again the next time you open the app.'**
  String get privacyRevokeConsentBody;

  /// No description provided for @privacyDeleteAccount.
  ///
  /// In en, this message translates to:
  /// **'Delete account and data'**
  String get privacyDeleteAccount;

  /// No description provided for @deleteAccountTitle.
  ///
  /// In en, this message translates to:
  /// **'Delete account?'**
  String get deleteAccountTitle;

  /// No description provided for @deleteAccountBody.
  ///
  /// In en, this message translates to:
  /// **'Permanently deletes your account, history and conversations on the server, plus the health profile and reminders on this device. This cannot be undone.'**
  String get deleteAccountBody;

  /// No description provided for @deleteAccountPasswordLabel.
  ///
  /// In en, this message translates to:
  /// **'Enter your password to confirm'**
  String get deleteAccountPasswordLabel;

  /// No description provided for @deleteAccountConfirm.
  ///
  /// In en, this message translates to:
  /// **'Delete permanently'**
  String get deleteAccountConfirm;

  /// No description provided for @deleteAccountDone.
  ///
  /// In en, this message translates to:
  /// **'Account and data deleted.'**
  String get deleteAccountDone;

  /// No description provided for @chatAiLabel.
  ///
  /// In en, this message translates to:
  /// **'AI assistant · answers only from loaded sources'**
  String get chatAiLabel;

  /// No description provided for @navHome.
  ///
  /// In en, this message translates to:
  /// **'Home'**
  String get navHome;

  /// No description provided for @navReminders.
  ///
  /// In en, this message translates to:
  /// **'Reminders'**
  String get navReminders;

  /// No description provided for @navAssistant.
  ///
  /// In en, this message translates to:
  /// **'Assistant'**
  String get navAssistant;

  /// No description provided for @navProfile.
  ///
  /// In en, this message translates to:
  /// **'Profile'**
  String get navProfile;

  /// No description provided for @scanMedicine.
  ///
  /// In en, this message translates to:
  /// **'Scan medicine'**
  String get scanMedicine;

  /// No description provided for @homeGreeting.
  ///
  /// In en, this message translates to:
  /// **'Hello!'**
  String get homeGreeting;

  /// No description provided for @homeSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Here is your medicine for today'**
  String get homeSubtitle;

  /// No description provided for @homeTodayTitle.
  ///
  /// In en, this message translates to:
  /// **'Today\'s medicines'**
  String get homeTodayTitle;

  /// No description provided for @homeNoReminders.
  ///
  /// In en, this message translates to:
  /// **'No reminders yet. Add one under Reminders.'**
  String get homeNoReminders;

  /// No description provided for @homeSeeAll.
  ///
  /// In en, this message translates to:
  /// **'See all'**
  String get homeSeeAll;

  /// No description provided for @homeSearchHint.
  ///
  /// In en, this message translates to:
  /// **'Search medicine by name…'**
  String get homeSearchHint;

  /// No description provided for @homeShortcuts.
  ///
  /// In en, this message translates to:
  /// **'Shortcuts'**
  String get homeShortcuts;

  /// No description provided for @reminderSetAlarm.
  ///
  /// In en, this message translates to:
  /// **'Set your alarm'**
  String get reminderSetAlarm;

  /// No description provided for @reminderSave.
  ///
  /// In en, this message translates to:
  /// **'Save'**
  String get reminderSave;

  /// No description provided for @reminderEveryDay.
  ///
  /// In en, this message translates to:
  /// **'Every day'**
  String get reminderEveryDay;

  /// No description provided for @reminderTimeLabel.
  ///
  /// In en, this message translates to:
  /// **'Time'**
  String get reminderTimeLabel;

  /// No description provided for @voiceMicTooltip.
  ///
  /// In en, this message translates to:
  /// **'Speak to the assistant'**
  String get voiceMicTooltip;

  /// No description provided for @voiceListening.
  ///
  /// In en, this message translates to:
  /// **'Listening…'**
  String get voiceListening;

  /// No description provided for @voiceUnavailable.
  ///
  /// In en, this message translates to:
  /// **'Voice is not enabled'**
  String get voiceUnavailable;

  /// No description provided for @profileTitle.
  ///
  /// In en, this message translates to:
  /// **'My profile'**
  String get profileTitle;

  /// No description provided for @profileAccountSection.
  ///
  /// In en, this message translates to:
  /// **'Account'**
  String get profileAccountSection;

  /// No description provided for @profileDataSection.
  ///
  /// In en, this message translates to:
  /// **'Data & privacy'**
  String get profileDataSection;

  /// No description provided for @homeGreetingNamed.
  ///
  /// In en, this message translates to:
  /// **'Hello, {name}!'**
  String homeGreetingNamed(String name);

  /// No description provided for @homeMarkTaken.
  ///
  /// In en, this message translates to:
  /// **'Mark as taken'**
  String get homeMarkTaken;

  /// No description provided for @homeMarkedTaken.
  ///
  /// In en, this message translates to:
  /// **'Taken today'**
  String get homeMarkedTaken;

  /// No description provided for @doseASkip.
  ///
  /// In en, this message translates to:
  /// **'Skip'**
  String get doseASkip;

  /// No description provided for @doseASnooze.
  ///
  /// In en, this message translates to:
  /// **'Snooze 15 min'**
  String get doseASnooze;

  /// No description provided for @doseASkippedToday.
  ///
  /// In en, this message translates to:
  /// **'Skipped today'**
  String get doseASkippedToday;

  /// No description provided for @doseANextDose.
  ///
  /// In en, this message translates to:
  /// **'Next dose'**
  String get doseANextDose;

  /// No description provided for @doseAMorning.
  ///
  /// In en, this message translates to:
  /// **'Morning'**
  String get doseAMorning;

  /// No description provided for @doseANoon.
  ///
  /// In en, this message translates to:
  /// **'Noon'**
  String get doseANoon;

  /// No description provided for @doseAAfternoon.
  ///
  /// In en, this message translates to:
  /// **'Afternoon'**
  String get doseAAfternoon;

  /// No description provided for @doseAEvening.
  ///
  /// In en, this message translates to:
  /// **'Evening'**
  String get doseAEvening;

  /// No description provided for @doseASnoozedMsg.
  ///
  /// In en, this message translates to:
  /// **'Will remind you again in 15 minutes'**
  String get doseASnoozedMsg;

  /// No description provided for @progressShortcut.
  ///
  /// In en, this message translates to:
  /// **'Progress'**
  String get progressShortcut;

  /// No description provided for @progressTitle.
  ///
  /// In en, this message translates to:
  /// **'Adherence progress'**
  String get progressTitle;

  /// No description provided for @progressStreakDays.
  ///
  /// In en, this message translates to:
  /// **'{count}-day streak!'**
  String progressStreakDays(int count);

  /// No description provided for @progressStreakZero.
  ///
  /// In en, this message translates to:
  /// **'Take your medicine today to start a streak'**
  String get progressStreakZero;

  /// No description provided for @progressWeekTitle.
  ///
  /// In en, this message translates to:
  /// **'Last 7 days'**
  String get progressWeekTitle;

  /// No description provided for @progressEmpty.
  ///
  /// In en, this message translates to:
  /// **'No doses logged yet.'**
  String get progressEmpty;

  /// No description provided for @reminderDuplicateWarning.
  ///
  /// In en, this message translates to:
  /// **'This looks like a medicine you already have in reminders or your health profile — double-check before saving.'**
  String get reminderDuplicateWarning;

  /// No description provided for @lockSectionTitle.
  ///
  /// In en, this message translates to:
  /// **'Lock app with a PIN'**
  String get lockSectionTitle;

  /// No description provided for @lockSectionSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Require a PIN when reopening the app after being away for a while.'**
  String get lockSectionSubtitle;

  /// No description provided for @lockEnterPinTitle.
  ///
  /// In en, this message translates to:
  /// **'Enter your PIN to unlock'**
  String get lockEnterPinTitle;

  /// No description provided for @lockUnlock.
  ///
  /// In en, this message translates to:
  /// **'Unlock'**
  String get lockUnlock;

  /// No description provided for @lockPinWrong.
  ///
  /// In en, this message translates to:
  /// **'Wrong PIN, try again.'**
  String get lockPinWrong;

  /// No description provided for @lockSetNewPinTitle.
  ///
  /// In en, this message translates to:
  /// **'Set a new PIN'**
  String get lockSetNewPinTitle;

  /// No description provided for @lockPinLabel.
  ///
  /// In en, this message translates to:
  /// **'PIN (4-6 digits)'**
  String get lockPinLabel;

  /// No description provided for @lockPinConfirmLabel.
  ///
  /// In en, this message translates to:
  /// **'Confirm PIN'**
  String get lockPinConfirmLabel;

  /// No description provided for @lockPinTooShort.
  ///
  /// In en, this message translates to:
  /// **'PIN must be at least 4 digits.'**
  String get lockPinTooShort;

  /// No description provided for @lockPinMismatch.
  ///
  /// In en, this message translates to:
  /// **'The two PINs don\'t match.'**
  String get lockPinMismatch;

  /// No description provided for @lockDisableTitle.
  ///
  /// In en, this message translates to:
  /// **'Enter your current PIN to turn off the lock'**
  String get lockDisableTitle;

  /// No description provided for @lockSave.
  ///
  /// In en, this message translates to:
  /// **'Save'**
  String get lockSave;

  /// No description provided for @lockEnabled.
  ///
  /// In en, this message translates to:
  /// **'PIN lock enabled'**
  String get lockEnabled;

  /// No description provided for @lockDisabled.
  ///
  /// In en, this message translates to:
  /// **'PIN lock disabled'**
  String get lockDisabled;

  /// No description provided for @newsLeaveTitle.
  ///
  /// In en, this message translates to:
  /// **'Leave the app to read this?'**
  String get newsLeaveTitle;

  /// No description provided for @newsLeaveBody.
  ///
  /// In en, this message translates to:
  /// **'The article opens in your browser at {host}. That site is not run by MedAssist and can see your IP address. No health data is sent.'**
  String newsLeaveBody(String host);

  /// No description provided for @newsVerifiedSources.
  ///
  /// In en, this message translates to:
  /// **'Links only from verified sources: {domains}'**
  String newsVerifiedSources(String domains);

  /// No description provided for @newsLeaveOpen.
  ///
  /// In en, this message translates to:
  /// **'Open browser'**
  String get newsLeaveOpen;

  /// No description provided for @newsLeaveCopy.
  ///
  /// In en, this message translates to:
  /// **'Copy link'**
  String get newsLeaveCopy;

  /// No description provided for @rxCreateFromScan.
  ///
  /// In en, this message translates to:
  /// **'Create reminders from prescription'**
  String get rxCreateFromScan;

  /// No description provided for @rxNothingFound.
  ///
  /// In en, this message translates to:
  /// **'No medicines found in this prescription. Please add reminders manually.'**
  String get rxNothingFound;

  /// No description provided for @rxReviewTitle.
  ///
  /// In en, this message translates to:
  /// **'Confirm prescription'**
  String get rxReviewTitle;

  /// No description provided for @rxAiBanner.
  ///
  /// In en, this message translates to:
  /// **'AI read this prescription — check everything before saving; handwritten prescriptions may be misread.'**
  String get rxAiBanner;

  /// No description provided for @rxInclude.
  ///
  /// In en, this message translates to:
  /// **'Add to reminders'**
  String get rxInclude;

  /// No description provided for @rxMedicineName.
  ///
  /// In en, this message translates to:
  /// **'Medicine name'**
  String get rxMedicineName;

  /// No description provided for @rxDose.
  ///
  /// In en, this message translates to:
  /// **'Dose per time'**
  String get rxDose;

  /// No description provided for @rxTimes.
  ///
  /// In en, this message translates to:
  /// **'Times'**
  String get rxTimes;

  /// No description provided for @rxAddTime.
  ///
  /// In en, this message translates to:
  /// **'Add time'**
  String get rxAddTime;

  /// No description provided for @rxMealRelation.
  ///
  /// In en, this message translates to:
  /// **'With meals'**
  String get rxMealRelation;

  /// No description provided for @rxMealNone.
  ///
  /// In en, this message translates to:
  /// **'Not specified'**
  String get rxMealNone;

  /// No description provided for @rxMealBefore.
  ///
  /// In en, this message translates to:
  /// **'Before meals'**
  String get rxMealBefore;

  /// No description provided for @rxMealAfter.
  ///
  /// In en, this message translates to:
  /// **'After meals'**
  String get rxMealAfter;

  /// No description provided for @rxMealDuring.
  ///
  /// In en, this message translates to:
  /// **'During meals'**
  String get rxMealDuring;

  /// No description provided for @rxNeedsReview.
  ///
  /// In en, this message translates to:
  /// **'Check this: frequency could not be read clearly'**
  String get rxNeedsReview;

  /// No description provided for @rxDuplicate.
  ///
  /// In en, this message translates to:
  /// **'This medicine is already in your reminders'**
  String get rxDuplicate;

  /// No description provided for @rxDiagnosisTitle.
  ///
  /// In en, this message translates to:
  /// **'Diagnosis (read from prescription)'**
  String get rxDiagnosisTitle;

  /// No description provided for @rxAddDiagnosisToProfile.
  ///
  /// In en, this message translates to:
  /// **'Add to health profile'**
  String get rxAddDiagnosisToProfile;

  /// No description provided for @rxProfileConsentRequired.
  ///
  /// In en, this message translates to:
  /// **'Health profile consent is required before adding the diagnosis or medicines to your profile.'**
  String get rxProfileConsentRequired;

  /// No description provided for @rxSave.
  ///
  /// In en, this message translates to:
  /// **'Save and create reminders'**
  String get rxSave;

  /// No description provided for @rxSaved.
  ///
  /// In en, this message translates to:
  /// **'Created {count} reminders'**
  String rxSaved(int count);

  /// No description provided for @rxNoItemSelected.
  ///
  /// In en, this message translates to:
  /// **'Select at least one medicine with a name and a time.'**
  String get rxNoItemSelected;

  /// No description provided for @rxSaveFailed.
  ///
  /// In en, this message translates to:
  /// **'Could not save the prescription. Please try again.'**
  String get rxSaveFailed;

  /// No description provided for @facilityTitle.
  ///
  /// In en, this message translates to:
  /// **'Nearest medical facility'**
  String get facilityTitle;

  /// No description provided for @facilityCall115.
  ///
  /// In en, this message translates to:
  /// **'Call emergency 115'**
  String get facilityCall115;

  /// No description provided for @facilitySampleBanner.
  ///
  /// In en, this message translates to:
  /// **'Sample data, needs verification — confirm the address before going.'**
  String get facilitySampleBanner;

  /// No description provided for @facilityLocationRationale.
  ///
  /// In en, this message translates to:
  /// **'Your location is used only to calculate distances on this phone. It is never sent anywhere.'**
  String get facilityLocationRationale;

  /// No description provided for @facilityUseLocation.
  ///
  /// In en, this message translates to:
  /// **'Use my location'**
  String get facilityUseLocation;

  /// No description provided for @facilityLocating.
  ///
  /// In en, this message translates to:
  /// **'Getting your location…'**
  String get facilityLocating;

  /// No description provided for @facilityPermissionDenied.
  ///
  /// In en, this message translates to:
  /// **'Location permission was denied. The list is not sorted by distance; you can grant permission in Settings.'**
  String get facilityPermissionDenied;

  /// No description provided for @facilityLocationOff.
  ///
  /// In en, this message translates to:
  /// **'Location is turned off. Turn on Location and try again.'**
  String get facilityLocationOff;

  /// No description provided for @facilityLocationUnavailable.
  ///
  /// In en, this message translates to:
  /// **'Could not get your location. Try again or search on the map.'**
  String get facilityLocationUnavailable;

  /// No description provided for @facilityUnsupported.
  ///
  /// In en, this message translates to:
  /// **'This device does not support location features.'**
  String get facilityUnsupported;

  /// No description provided for @facilityDirections.
  ///
  /// In en, this message translates to:
  /// **'Directions'**
  String get facilityDirections;

  /// No description provided for @facilitySearchMore.
  ///
  /// In en, this message translates to:
  /// **'Search more on the map'**
  String get facilitySearchMore;

  /// No description provided for @facilitySearchQuery.
  ///
  /// In en, this message translates to:
  /// **'hospitals near me'**
  String get facilitySearchQuery;

  /// No description provided for @facilityDistanceKm.
  ///
  /// In en, this message translates to:
  /// **'{km} km away'**
  String facilityDistanceKm(String km);

  /// No description provided for @facilityOpenFailed.
  ///
  /// In en, this message translates to:
  /// **'No app on this phone can handle this action.'**
  String get facilityOpenFailed;

  /// No description provided for @facilityLoadFailed.
  ///
  /// In en, this message translates to:
  /// **'Could not load the medical facility list.'**
  String get facilityLoadFailed;

  /// No description provided for @cloudAiTitle.
  ///
  /// In en, this message translates to:
  /// **'Cloud AI (Claude)'**
  String get cloudAiTitle;

  /// No description provided for @cloudAiSubtitle.
  ///
  /// In en, this message translates to:
  /// **'Assistant answers more naturally using Anthropic\'s AI. Off by default.'**
  String get cloudAiSubtitle;

  /// No description provided for @cloudAiDialogTitle.
  ///
  /// In en, this message translates to:
  /// **'Use cloud AI?'**
  String get cloudAiDialogTitle;

  /// No description provided for @cloudAiDialogBody.
  ///
  /// In en, this message translates to:
  /// **'Recipient: Anthropic (Claude, cloud AI).\n\nSent: your question (with identifying details such as name, phone, email, dates and document numbers removed), the medicines you currently take, and the conditions/allergies you entered in your profile.\n\nNote: for \"Tips\", medicine and condition names go to the MedAssist server for that request only (not stored, not logged) - separate from cloud AI.\n\nNot sent: prescription photos, location, your full profile.\n\nYou can turn this off at any time. Information is for reference only and is not a diagnosis.'**
  String get cloudAiDialogBody;

  /// No description provided for @cloudAiAgree.
  ///
  /// In en, this message translates to:
  /// **'Agree'**
  String get cloudAiAgree;

  /// No description provided for @cloudAiDecline.
  ///
  /// In en, this message translates to:
  /// **'No'**
  String get cloudAiDecline;

  /// No description provided for @cloudAiGrantedAt.
  ///
  /// In en, this message translates to:
  /// **'Agreed on {when}'**
  String cloudAiGrantedAt(String when);

  /// No description provided for @cloudAiChatHint.
  ///
  /// In en, this message translates to:
  /// **'Turn on \'Cloud AI\' in Profile > Privacy for more natural answers'**
  String get cloudAiChatHint;

  /// No description provided for @cloudAiHintDismiss.
  ///
  /// In en, this message translates to:
  /// **'Dismiss hint'**
  String get cloudAiHintDismiss;

  /// No description provided for @emergencyCall.
  ///
  /// In en, this message translates to:
  /// **'Call 115'**
  String get emergencyCall;

  /// No description provided for @emergencyNearest.
  ///
  /// In en, this message translates to:
  /// **'Nearest medical facility'**
  String get emergencyNearest;

  /// No description provided for @emergencyDialFailed.
  ///
  /// In en, this message translates to:
  /// **'Could not open the dialer. Please call 115 manually.'**
  String get emergencyDialFailed;

  /// No description provided for @lifestyleTitle.
  ///
  /// In en, this message translates to:
  /// **'Tips for you'**
  String get lifestyleTitle;

  /// No description provided for @lifestyleEmpty.
  ///
  /// In en, this message translates to:
  /// **'No tips right now.'**
  String get lifestyleEmpty;

  /// No description provided for @lifestyleError.
  ///
  /// In en, this message translates to:
  /// **'Could not load tips.'**
  String get lifestyleError;

  /// No description provided for @lifestyleRefresh.
  ///
  /// In en, this message translates to:
  /// **'Refresh tips'**
  String get lifestyleRefresh;

  /// No description provided for @lifestyleSource.
  ///
  /// In en, this message translates to:
  /// **'Source: {name}'**
  String lifestyleSource(String name);

  /// No description provided for @lifestyleAiBadge.
  ///
  /// In en, this message translates to:
  /// **'AI'**
  String get lifestyleAiBadge;

  /// No description provided for @polishDViewMedicine.
  ///
  /// In en, this message translates to:
  /// **'View uses & side effects'**
  String get polishDViewMedicine;

  /// No description provided for @polishDQuoteLabel.
  ///
  /// In en, this message translates to:
  /// **'Excerpt ({id})'**
  String polishDQuoteLabel(String id);

  /// No description provided for @stockBPillsTotal.
  ///
  /// In en, this message translates to:
  /// **'Pills dispensed (optional)'**
  String get stockBPillsTotal;

  /// No description provided for @stockBPillsLeft.
  ///
  /// In en, this message translates to:
  /// **'Pills left (optional)'**
  String get stockBPillsLeft;

  /// No description provided for @stockBPillsPerDose.
  ///
  /// In en, this message translates to:
  /// **'Pills per dose'**
  String get stockBPillsPerDose;

  /// No description provided for @stockBConfirmed.
  ///
  /// In en, this message translates to:
  /// **'I have checked this'**
  String get stockBConfirmed;

  /// No description provided for @stockBNeedsCheck.
  ///
  /// In en, this message translates to:
  /// **'Please check this item against the prescription and tick the box.'**
  String get stockBNeedsCheck;

  /// No description provided for @stockBLowStock.
  ///
  /// In en, this message translates to:
  /// **'Running low: {name} has {count} pills left. Buy more or ask a pharmacist.'**
  String stockBLowStock(String name, int count);

  /// No description provided for @stockBEmptyReminderTitle.
  ///
  /// In en, this message translates to:
  /// **'No reminders yet'**
  String get stockBEmptyReminderTitle;

  /// No description provided for @stockBScanPrescription.
  ///
  /// In en, this message translates to:
  /// **'Scan prescription'**
  String get stockBScanPrescription;

  /// No description provided for @reportCTitle.
  ///
  /// In en, this message translates to:
  /// **'Adherence report'**
  String get reportCTitle;

  /// No description provided for @reportCRange7.
  ///
  /// In en, this message translates to:
  /// **'7 days'**
  String get reportCRange7;

  /// No description provided for @reportCRange30.
  ///
  /// In en, this message translates to:
  /// **'30 days'**
  String get reportCRange30;

  /// No description provided for @reportCRate.
  ///
  /// In en, this message translates to:
  /// **'On-time rate'**
  String get reportCRate;

  /// No description provided for @reportCTaken.
  ///
  /// In en, this message translates to:
  /// **'Taken'**
  String get reportCTaken;

  /// No description provided for @reportCSkipped.
  ///
  /// In en, this message translates to:
  /// **'Skipped'**
  String get reportCSkipped;

  /// No description provided for @reportCLate.
  ///
  /// In en, this message translates to:
  /// **'Late'**
  String get reportCLate;

  /// No description provided for @reportCSnoozed.
  ///
  /// In en, this message translates to:
  /// **'Snoozed'**
  String get reportCSnoozed;

  /// No description provided for @reportCStreakNow.
  ///
  /// In en, this message translates to:
  /// **'Current streak'**
  String get reportCStreakNow;

  /// No description provided for @reportCStreakBest.
  ///
  /// In en, this message translates to:
  /// **'Best streak'**
  String get reportCStreakBest;

  /// No description provided for @reportCDays.
  ///
  /// In en, this message translates to:
  /// **'days'**
  String get reportCDays;

  /// No description provided for @reportCChartTitle.
  ///
  /// In en, this message translates to:
  /// **'Doses taken per day'**
  String get reportCChartTitle;

  /// No description provided for @reportCNoRate.
  ///
  /// In en, this message translates to:
  /// **'No data'**
  String get reportCNoRate;

  /// No description provided for @reportCNote.
  ///
  /// In en, this message translates to:
  /// **'For reference only; talk to your doctor or pharmacist. Do not change or stop medicine on your own.'**
  String get reportCNote;

  /// No description provided for @reportCDoctorMode.
  ///
  /// In en, this message translates to:
  /// **'Show to doctor'**
  String get reportCDoctorMode;

  /// No description provided for @reportCDoctorTitle.
  ///
  /// In en, this message translates to:
  /// **'Summary for doctor'**
  String get reportCDoctorTitle;

  /// No description provided for @reportCBack.
  ///
  /// In en, this message translates to:
  /// **'Back'**
  String get reportCBack;

  /// No description provided for @reportCEmpty.
  ///
  /// In en, this message translates to:
  /// **'No doses logged yet. Mark doses as taken to see your report.'**
  String get reportCEmpty;

  /// No description provided for @reportCDoctorHint.
  ///
  /// In en, this message translates to:
  /// **'Read-only summary stored only on this phone.'**
  String get reportCDoctorHint;

  /// No description provided for @reportCPeriod.
  ///
  /// In en, this message translates to:
  /// **'Period'**
  String get reportCPeriod;

  /// No description provided for @fixELifestyleNote.
  ///
  /// In en, this message translates to:
  /// **'For \"Tips\", medicine and condition names are sent to the MedAssist server for that request only (not stored, not logged). This is separate from sending data to cloud AI.'**
  String get fixELifestyleNote;

  /// No description provided for @fixELockTooMany.
  ///
  /// In en, this message translates to:
  /// **'Too many wrong attempts. Try again in {time}.'**
  String fixELockTooMany(String time);

  /// No description provided for @fixELockForgot.
  ///
  /// In en, this message translates to:
  /// **'Forgot PIN? Sign out'**
  String get fixELockForgot;

  /// No description provided for @fixELockUnreadable.
  ///
  /// In en, this message translates to:
  /// **'Secure storage cannot be read, so the app stays locked. Sign out to reset the PIN.'**
  String get fixELockUnreadable;

  /// No description provided for @modeGUnverified.
  ///
  /// In en, this message translates to:
  /// **'Sample data - not verified'**
  String get modeGUnverified;

  /// No description provided for @modeGUnmatched.
  ///
  /// In en, this message translates to:
  /// **'No verified nutrition document for: {conditions}. Please ask a doctor or a dietitian.'**
  String modeGUnmatched(String conditions);

  /// No description provided for @modeGChipExtractive.
  ///
  /// In en, this message translates to:
  /// **'Quoted from loaded sources'**
  String get modeGChipExtractive;

  /// No description provided for @modeGChipCloud.
  ///
  /// In en, this message translates to:
  /// **'Reworded by Claude - still source-based'**
  String get modeGChipCloud;

  /// No description provided for @modeGSheetTitle.
  ///
  /// In en, this message translates to:
  /// **'Two answer modes'**
  String get modeGSheetTitle;

  /// No description provided for @modeGOff.
  ///
  /// In en, this message translates to:
  /// **'Cloud AI off'**
  String get modeGOff;

  /// No description provided for @modeGOn.
  ///
  /// In en, this message translates to:
  /// **'Cloud AI on'**
  String get modeGOn;

  /// No description provided for @modeGRowData.
  ///
  /// In en, this message translates to:
  /// **'Does data leave the MedAssist server?'**
  String get modeGRowData;

  /// No description provided for @modeGOffData.
  ///
  /// In en, this message translates to:
  /// **'No. Your question is handled on the MedAssist server only, nothing goes to a third party.'**
  String get modeGOffData;

  /// No description provided for @modeGOnData.
  ///
  /// In en, this message translates to:
  /// **'Yes. Your question (identifying details removed) plus the medicines/conditions you agreed to share go to Anthropic (Claude).'**
  String get modeGOnData;

  /// No description provided for @modeGRowHow.
  ///
  /// In en, this message translates to:
  /// **'How it answers'**
  String get modeGRowHow;

  /// No description provided for @modeGOffHow.
  ///
  /// In en, this message translates to:
  /// **'Quotes the loaded documents verbatim.'**
  String get modeGOffHow;

  /// No description provided for @modeGOnHow.
  ///
  /// In en, this message translates to:
  /// **'Claude rewords it more naturally using only the loaded documents; if it strays from the sources, the verbatim quote is used instead.'**
  String get modeGOnHow;

  /// No description provided for @modeGRowSame.
  ///
  /// In en, this message translates to:
  /// **'What does NOT change'**
  String get modeGRowSame;

  /// No description provided for @modeGSameBody.
  ///
  /// In en, this message translates to:
  /// **'Sources are always cited; no diagnosis or dosing; emergency warnings still appear.'**
  String get modeGSameBody;

  /// No description provided for @modeGWhere.
  ///
  /// In en, this message translates to:
  /// **'Want to switch? Go to Profile > Privacy > Cloud AI.'**
  String get modeGWhere;

  /// No description provided for @modeGClose.
  ///
  /// In en, this message translates to:
  /// **'Got it'**
  String get modeGClose;

  /// No description provided for @modeGConsentModes.
  ///
  /// In en, this message translates to:
  /// **'Off: answers quote the loaded documents verbatim and nothing is sent to a third party. On: Claude (Anthropic) rewords answers more naturally but must still follow the sources.'**
  String get modeGConsentModes;

  /// No description provided for @memHTitle.
  ///
  /// In en, this message translates to:
  /// **'What the assistant remembers about me'**
  String get memHTitle;

  /// No description provided for @memHSubtitle.
  ///
  /// In en, this message translates to:
  /// **'See, turn off or delete what the assistant remembers - stored on this device only.'**
  String get memHSubtitle;

  /// No description provided for @memHToggle.
  ///
  /// In en, this message translates to:
  /// **'Let the assistant remember my symptoms'**
  String get memHToggle;

  /// No description provided for @memHToggleBody.
  ///
  /// In en, this message translates to:
  /// **'Stored ONLY on this device, encrypted, and you can delete it any time. Each question sends only the minimum context to the MedAssist server, which does not store it.'**
  String get memHToggleBody;

  /// No description provided for @memHNeedProfile.
  ///
  /// In en, this message translates to:
  /// **'Turn on the Health profile (consent to store) first.'**
  String get memHNeedProfile;

  /// No description provided for @memHUnreadable.
  ///
  /// In en, this message translates to:
  /// **'Secure data cannot be read'**
  String get memHUnreadable;

  /// No description provided for @memHUnreadableBody.
  ///
  /// In en, this message translates to:
  /// **'The encrypted data on this device cannot be read. The app will not overwrite it. You can delete all assistant data to start over.'**
  String get memHUnreadableBody;

  /// No description provided for @memHStoredTitle.
  ///
  /// In en, this message translates to:
  /// **'Stored on this device'**
  String get memHStoredTitle;

  /// No description provided for @memHConditions.
  ///
  /// In en, this message translates to:
  /// **'Conditions: {count}'**
  String memHConditions(int count);

  /// No description provided for @memHAllergies.
  ///
  /// In en, this message translates to:
  /// **'Allergies: {count}'**
  String memHAllergies(int count);

  /// No description provided for @memHMeds.
  ///
  /// In en, this message translates to:
  /// **'Current medications: {count}'**
  String memHMeds(int count);

  /// No description provided for @memHEpisodes.
  ///
  /// In en, this message translates to:
  /// **'Symptom log: {count} entries'**
  String memHEpisodes(int count);

  /// No description provided for @memHNone.
  ///
  /// In en, this message translates to:
  /// **'None'**
  String get memHNone;

  /// No description provided for @memHEpisodesTitle.
  ///
  /// In en, this message translates to:
  /// **'Last 20 entries'**
  String get memHEpisodesTitle;

  /// No description provided for @memHRecurringTag.
  ///
  /// In en, this message translates to:
  /// **'recurring'**
  String get memHRecurringTag;

  /// No description provided for @memHPreviewTitle.
  ///
  /// In en, this message translates to:
  /// **'What the assistant will use for your next question'**
  String get memHPreviewTitle;

  /// No description provided for @memHPreviewConditions.
  ///
  /// In en, this message translates to:
  /// **'Conditions'**
  String get memHPreviewConditions;

  /// No description provided for @memHPreviewAllergies.
  ///
  /// In en, this message translates to:
  /// **'Allergies'**
  String get memHPreviewAllergies;

  /// No description provided for @memHPreviewSymptoms.
  ///
  /// In en, this message translates to:
  /// **'Recent symptoms'**
  String get memHPreviewSymptoms;

  /// No description provided for @memHPreviewNothing.
  ///
  /// In en, this message translates to:
  /// **'Nothing is sent besides your question.'**
  String get memHPreviewNothing;

  /// No description provided for @memHCloudYes.
  ///
  /// In en, this message translates to:
  /// **'Cloud AI: receives only the de-identified question (you consented separately).'**
  String get memHCloudYes;

  /// No description provided for @memHCloudNo.
  ///
  /// In en, this message translates to:
  /// **'Cloud AI: receives nothing (no consent given).'**
  String get memHCloudNo;

  /// No description provided for @memHClearLog.
  ///
  /// In en, this message translates to:
  /// **'Delete symptom log'**
  String get memHClearLog;

  /// No description provided for @memHClearAll.
  ///
  /// In en, this message translates to:
  /// **'Delete all assistant data'**
  String get memHClearAll;

  /// No description provided for @memHClearLogConfirm.
  ///
  /// In en, this message translates to:
  /// **'Delete the whole symptom log on this device?'**
  String get memHClearLogConfirm;

  /// No description provided for @memHClearAllConfirm.
  ///
  /// In en, this message translates to:
  /// **'Delete the health profile, symptom log and turn off assistant memory on this device? This cannot be undone.'**
  String get memHClearAllConfirm;

  /// No description provided for @memHCleared.
  ///
  /// In en, this message translates to:
  /// **'Deleted'**
  String get memHCleared;

  /// No description provided for @memHChatSaved.
  ///
  /// In en, this message translates to:
  /// **'Remembered on this device: {terms}'**
  String memHChatSaved(String terms);

  /// No description provided for @memHChatForget.
  ///
  /// In en, this message translates to:
  /// **'Delete'**
  String get memHChatForget;

  /// No description provided for @memHRecurring.
  ///
  /// In en, this message translates to:
  /// **'Recurring symptom'**
  String get memHRecurring;

  /// No description provided for @srcJTitle.
  ///
  /// In en, this message translates to:
  /// **'Source'**
  String get srcJTitle;

  /// No description provided for @srcJRetrieved.
  ///
  /// In en, this message translates to:
  /// **'Retrieved: {date}'**
  String srcJRetrieved(String date);

  /// No description provided for @srcJLicense.
  ///
  /// In en, this message translates to:
  /// **'License: {text}'**
  String srcJLicense(String text);

  /// No description provided for @srcJReviewed.
  ///
  /// In en, this message translates to:
  /// **'Reviewed by: {name}'**
  String srcJReviewed(String name);

  /// No description provided for @srcJNotReviewed.
  ///
  /// In en, this message translates to:
  /// **'No reviewer yet'**
  String get srcJNotReviewed;

  /// No description provided for @srcJOpen.
  ///
  /// In en, this message translates to:
  /// **'Open original source'**
  String get srcJOpen;

  /// No description provided for @srcJOpenHost.
  ///
  /// In en, this message translates to:
  /// **'Open original source ({host})'**
  String srcJOpenHost(String host);

  /// No description provided for @srcJNoLink.
  ///
  /// In en, this message translates to:
  /// **'No original link'**
  String get srcJNoLink;

  /// No description provided for @srcJVerified.
  ///
  /// In en, this message translates to:
  /// **'Verified source'**
  String get srcJVerified;

  /// No description provided for @rxCheckCatalog.
  ///
  /// In en, this message translates to:
  /// **'Check against medicine list'**
  String get rxCheckCatalog;

  /// No description provided for @rxCheckCatalogNote.
  ///
  /// In en, this message translates to:
  /// **'Sends the medicine name to the MedAssist server to look it up in the list (not stored, not logged).'**
  String get rxCheckCatalogNote;

  /// No description provided for @rxCatalogExact.
  ///
  /// In en, this message translates to:
  /// **'Found in the medicine list'**
  String get rxCatalogExact;

  /// No description provided for @rxCatalogSimilar.
  ///
  /// In en, this message translates to:
  /// **'No exact match — see suggestions below and check the spelling'**
  String get rxCatalogSimilar;

  /// No description provided for @rxCatalogNone.
  ///
  /// In en, this message translates to:
  /// **'Not found in the medicine list — OCR may have misread it; compare with the prescription'**
  String get rxCatalogNone;

  /// No description provided for @rxCatalogError.
  ///
  /// In en, this message translates to:
  /// **'Could not check (offline?). Please verify manually.'**
  String get rxCatalogError;

  /// No description provided for @rxUseThisName.
  ///
  /// In en, this message translates to:
  /// **'Use this name'**
  String get rxUseThisName;

  /// No description provided for @rxCatalogStrength.
  ///
  /// In en, this message translates to:
  /// **'Ingredient · strength: {value}'**
  String rxCatalogStrength(String value);

  /// No description provided for @rxLowConfidence.
  ///
  /// In en, this message translates to:
  /// **'Low confidence: {reasons}'**
  String rxLowConfidence(String reasons);

  /// No description provided for @rxIssueFrequency.
  ///
  /// In en, this message translates to:
  /// **'frequency unclear'**
  String get rxIssueFrequency;

  /// No description provided for @rxIssueDose.
  ///
  /// In en, this message translates to:
  /// **'dose missing'**
  String get rxIssueDose;

  /// No description provided for @rxIssueShortName.
  ///
  /// In en, this message translates to:
  /// **'name too short'**
  String get rxIssueShortName;

  /// No description provided for @rxIssueOddChars.
  ///
  /// In en, this message translates to:
  /// **'name has unusual characters'**
  String get rxIssueOddChars;

  /// No description provided for @rxIssueNotInCatalog.
  ///
  /// In en, this message translates to:
  /// **'not in the medicine list'**
  String get rxIssueNotInCatalog;

  /// No description provided for @displaySection.
  ///
  /// In en, this message translates to:
  /// **'Display'**
  String get displaySection;

  /// No description provided for @displayTheme.
  ///
  /// In en, this message translates to:
  /// **'Theme'**
  String get displayTheme;

  /// No description provided for @displayThemeSystem.
  ///
  /// In en, this message translates to:
  /// **'Device'**
  String get displayThemeSystem;

  /// No description provided for @displayThemeLight.
  ///
  /// In en, this message translates to:
  /// **'Light'**
  String get displayThemeLight;

  /// No description provided for @displayThemeDark.
  ///
  /// In en, this message translates to:
  /// **'Dark'**
  String get displayThemeDark;

  /// No description provided for @displayTextSize.
  ///
  /// In en, this message translates to:
  /// **'Text size'**
  String get displayTextSize;

  /// No description provided for @displayTextNormal.
  ///
  /// In en, this message translates to:
  /// **'Normal'**
  String get displayTextNormal;

  /// No description provided for @displayTextLarge.
  ///
  /// In en, this message translates to:
  /// **'Large'**
  String get displayTextLarge;

  /// No description provided for @displayTextExtraLarge.
  ///
  /// In en, this message translates to:
  /// **'Extra large'**
  String get displayTextExtraLarge;

  /// No description provided for @doseASnoozeMenu.
  ///
  /// In en, this message translates to:
  /// **'Snooze…'**
  String get doseASnoozeMenu;

  /// No description provided for @doseASnoozeFor.
  ///
  /// In en, this message translates to:
  /// **'Snooze {minutes} min'**
  String doseASnoozeFor(int minutes);

  /// No description provided for @doseASnoozedFor.
  ///
  /// In en, this message translates to:
  /// **'Will remind you again in {minutes} minutes'**
  String doseASnoozedFor(int minutes);
}

class _AppLocalizationsDelegate extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  Future<AppLocalizations> load(Locale locale) {
    return SynchronousFuture<AppLocalizations>(lookupAppLocalizations(locale));
  }

  @override
  bool isSupported(Locale locale) => <String>['en', 'vi'].contains(locale.languageCode);

  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}

AppLocalizations lookupAppLocalizations(Locale locale) {


  // Lookup logic when only language code is specified.
  switch (locale.languageCode) {
    case 'en': return AppLocalizationsEn();
    case 'vi': return AppLocalizationsVi();
  }

  throw FlutterError(
    'AppLocalizations.delegate failed to load unsupported locale "$locale". This is likely '
    'an issue with the localizations generation tool. Please file an issue '
    'on GitHub with a reproducible sample app and the gen-l10n configuration '
    'that was used.'
  );
}
