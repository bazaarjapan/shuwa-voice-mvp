import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://shuwa-voice-mvp.gigach2025.chatgpt.site'),
  title: 'しゅわボイス｜手の形をことばに',
  description: 'PCのカメラで手の形を認識し、文字と音声で伝える手話認識MVPです。',
  openGraph: {
    title: 'しゅわボイス｜手の形をことばに',
    description: 'PCのカメラで手の形を覚え、文字と音声で伝える手話認識MVPです。',
    images: [{ url: 'https://shuwa-voice-mvp.gigach2025.chatgpt.site/og.png', width: 1200, height: 630, alt: 'しゅわボイス — 手のかたちを、ことばに。' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'しゅわボイス｜手の形をことばに',
    description: 'PCのカメラで手の形を覚え、文字と音声で伝える手話認識MVPです。',
    images: ['https://shuwa-voice-mvp.gigach2025.chatgpt.site/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
